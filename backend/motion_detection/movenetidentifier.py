import numpy as np
import tensorflow as tf
import cv2
import time
from backend.support.support_func import find_distance, find_midpoint, find_angle, find_quad_angle
from backend.support.support_func import normalize, sigmoid
#helper functions

# Dictionary that maps from joint names to keypoint indices.
KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

# Maps bones to a matplotlib color name.
KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): (255,0,255),
    (0, 2): (0,255,255),
    (1, 3): (255,0,255),
    (2, 4): (0,255,255),
    (0, 5): (255,0,255),
    (0, 6): (0,255,255),
    (5, 7): (255,0,255),
    (7, 9): (255,0,255),
    (6, 8): (0,255,255),
    (8, 10): (0,255,255),
    (5, 6): (255,255,0),
    (5, 11): (255,0,255),
    (6, 12): (0,255,255),
    (11, 12): (255,255,0),
    (11, 13): (255,0,255),
    (13, 15): (255,0,255),
    (12, 14): (0,255,255),
    (14, 16): (0,255,255),
}

parts = ['nose','leye','reye','lear','rear','lshoulder',
        'rshoulder','lelbow','relbow','lwrist','rwrist','lhip','rhip','lknee',
        'rknee','lankle','rankle']

# angle of letter. first number is your left, second number is your right
LETTER_ANGLE = {
  'A': [-45, -90],
  'B': [0, -90],
  'C': [45, -90],
  'D': [90, -90],
  'E': [-90, 135],
  'F': [-90, -180],
  'G': [-90, -135],
  'H': [0, -45],
  'I': [45, -45],
  'J': [90, -180],
  'K': [-45, 90],
  'L': [-45, 135],
  'M': [-45, -180],
  'N': [-45, -135],
  'O': [0, 45],
  'P': [0, 90],
  'Q': [0, 135],
  'R': [0, -180],
  'S': [0, -135],
  'T': [45, 90],
  'U': [45, 135],
  'V': [90, -135],
  'W': [135, -180],
  'X': [135, -135],
  'Y': [45, -180],
  'Z': [-135, -180],
}

class MovenetIdentifier:
    def __init__(self,height, width, exercise_arg, threshold = 0.15, stickiness = 0.9, fugacity = 1.0):
        #print("Calling init MI")
        #width and height of displayimage
        self.width = width
        self.height = height

        #type of exercise
        self.exercise_arg = exercise_arg
        #unify later on
        self.squat_counter = 0
        self.squat_state_prev = 0

        #tuning parameters
        #stickiness increases tendency to remember previous entry
        #fugacity increases tendency to update to newest values
        self.stickiness = stickiness
        self.fugacity = fugacity


        #estimate of confidence of derived x,y
        self.confidence = np.zeros((17,1))
        #confidence threshold, confidence lower than this will be ignored.
        self.threshold = threshold

        #estimate of locations
        self.x = np.zeros((17,1))
        self.y = np.zeros((17,1))

        self.records = dict()
        self.records['x'] = dict()
        self.records['y'] = dict()


        for part in parts:
            self.records['x'][part] = []
            self.records['y'][part] = []

        self.records['time'] = []
        self.records['hip_angle'] = []
        self.records['knee_angle'] = []


    def update(self, current, img):
        #print("Calling update MI")
        #reshape input
        curr_confidence = current[:, 2]
        curr_confidence = np.reshape(curr_confidence,(17,1))

        x_prime = current[:, 1]
        x_prime = np.reshape(x_prime,(17,1))

        y_prime = current[:, 0]
        y_prime = np.reshape(y_prime,(17,1))

        #update estimate of x and y
        self.x = (self.x * self.confidence * self.stickiness + x_prime * curr_confidence * self.fugacity) / (curr_confidence * self.fugacity+ self.confidence * self.stickiness)
        self.y = (self.y * self.confidence * self.stickiness + y_prime * curr_confidence * self.fugacity) / (curr_confidence * self.fugacity+ self.confidence * self.stickiness)

        #update estimate of confidence
        #print("CONFIDENCE: ", self.confidence)
        self.confidence = (self.confidence ** 2 * self.stickiness + curr_confidence ** 2 * self.fugacity) / (self.confidence * self.stickiness+ curr_confidence * self.fugacity)
        for val_idx in range(17):
            if self.confidence[val_idx] > self.threshold:
                img = cv2.circle(img, (int(self.x[val_idx]),  int(self.y[val_idx])), 2, (0, 255, 0), 5)
                #img_sk = cv2.circle(img_sk, (int(self.x[val_idx]),  int(self.y[val_idx])), 2, (0, 255, 0), 5)

        img = self.draw_lines(img)

        #record values
        for part in parts:
            if self.confidence[parts.index(part)] > self.threshold:
                self.records['x'][part].append(self.x[parts.index(part)][0])
                self.records['y'][part].append(self.y[parts.index(part)][0])
            else:
                self.records['x'][part].append(None)
                self.records['y'][part].append(None)

        if self.exercise_arg == 1:
            # calculate squat angles
            if self.records['x']['lshoulder'][-1] and self.records['x']['lhip'][-1] and self.records['x']['lknee'][-1] and self.records['x']['lankle'][-1] and self.records['x']['nose'][-1]:
                lhiptoknee = find_distance(self.records['x']['lhip'][-1], self.records['x']['lknee'][-1], self.records['y']['lhip'][-1], self.records['y']['lknee'][-1])
                lkneetoankle = find_distance(self.records['x']['lknee'][-1], self.records['x']['lankle'][-1], self.records['y']['lknee'][-1], self.records['y']['lankle'][-1])
                lhiptoankle = find_distance(self.records['x']['lhip'][-1], self.records['x']['lankle'][-1], self.records['y']['lhip'][-1], self.records['y']['lankle'][-1])

                lshouldertohip = find_distance(self.records['x']['lshoulder'][-1], self.records['x']['lhip'][-1], self.records['y']['lshoulder'][-1], self.records['y']['lhip'][-1])
                lhiptoknee = find_distance(self.records['x']['lhip'][-1], self.records['x']['lknee'][-1], self.records['y']['lhip'][-1], self.records['y']['lknee'][-1])
                lshouldertoknee = find_distance(self.records['x']['lshoulder'][-1], self.records['x']['lknee'][-1], self.records['y']['lshoulder'][-1], self.records['y']['lknee'][-1])

                squatangleknee = 180 - int((180/np.pi) * find_angle(lhiptoknee, lkneetoankle, lhiptoankle))
                squatanglehip = 180 - int((180/np.pi) * find_angle(lshouldertohip, lhiptoknee, lshouldertoknee))



                cv2.putText(img, str(squatangleknee), org=(int(self.records['x']['lknee'][-1]) + 5, int(self.records['y']['lknee'][-1])), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 255, 0),thickness=3)
                cv2.putText(img, str(squatanglehip), org=(int(self.records['x']['lhip'][-1]) + 5, int(self.records['y']['lhip'][-1])), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 255, 0),thickness=3)

                dtonose = find_distance(self.records['x']['lhip'][-1], self.records['x']['nose'][-1], self.records['y']['lhip'][-1], self.records['y']['nose'][-1])
                dtofeet = find_distance(self.records['x']['lhip'][-1], self.records['x']['lankle'][-1], self.records['y']['lhip'][-1], self.records['y']['lankle'][-1])


            # check for squats
            if self.records['x']['lshoulder'][-1] and self.records['x']['lhip'][-1] and self.records['x']['lknee'][-1] and self.records['x']['lankle'][-1] and self.records['x']['nose'][-1]:
                lhiptoknee = find_distance(self.records['x']['lhip'][-1], self.records['x']['lknee'][-1], self.records['y']['lhip'][-1], self.records['y']['lknee'][-1])
                lkneetoankle = find_distance(self.records['x']['lknee'][-1], self.records['x']['lankle'][-1], self.records['y']['lknee'][-1], self.records['y']['lankle'][-1])
                lhiptoankle = find_distance(self.records['x']['lhip'][-1], self.records['x']['lankle'][-1], self.records['y']['lhip'][-1], self.records['y']['lankle'][-1])

                lshouldertohip = find_distance(self.records['x']['lshoulder'][-1], self.records['x']['lhip'][-1], self.records['y']['lshoulder'][-1], self.records['y']['lhip'][-1])
                lhiptoknee = find_distance(self.records['x']['lhip'][-1], self.records['x']['lknee'][-1], self.records['y']['lhip'][-1], self.records['y']['lknee'][-1])
                lshouldertoknee = find_distance(self.records['x']['lshoulder'][-1], self.records['x']['lknee'][-1], self.records['y']['lshoulder'][-1], self.records['y']['lknee'][-1])

                y_lhiptoknee = self.records['y']['lknee'][-1] - self.records['y']['lhip'][-1]
                y_lkneetoankle = self.records['y']['lankle'][-1] - self.records['y']['lknee'][-1]

                # check for squat
                # 0 = standing, 1 = bending, 2 = returning
                #was standing
                if self.squat_state_prev == 0:

                    if y_lhiptoknee < 0.2*y_lkneetoankle:
                        self.squat_state_prev = 1
                #was bending
                elif self.squat_state_prev == 1:
                    if y_lhiptoknee < 0.5*y_lkneetoankle:
                        self.squat_state_prev = 0
                        self.squat_counter += 1

                #get angles
                sak = int((180/np.pi) * find_angle(lhiptoknee, lkneetoankle, lhiptoankle))
                sah = int((180/np.pi) * find_angle(lshouldertohip, lhiptoknee, lshouldertoknee))

                self.records['knee_angle'].append(squatangleknee)
                self.records['hip_angle'].append(squatanglehip)
                ratio = squatangleknee/squatanglehip


                if sak <= 90:
                    squatangleknee = 180 - sak
                else:
                    squatangleknee = sak

                if sah <= 90:
                    squatanglehip = 180 - sah
                else:
                    squatanglehip = sah



        return img

    def draw_lines(self, img): #no change
        #img = cv2.line(img, (0,height//2), (width,height//2), (255,0,255), 3)
        #print("Calling draw MI")
        for pair in KEYPOINT_EDGE_INDS_TO_COLOR:
            start,end = pair
            if self.confidence[start]> self.threshold and self.confidence[end]> self.threshold:
                #print("confidence reached: ", self.confidence[start])
                img = cv2.line(img, (int(self.x[start]),int(self.y[start]))
                               , (int(self.x[end]),int(self.y[end]))
                               , KEYPOINT_EDGE_INDS_TO_COLOR[pair], 3)
                '''
                img_sk = cv2.line(img_sk, (int(self.x[start]),int(self.y[start]))
                               , (int(self.x[end]),int(self.y[end]))
                               , KEYPOINT_EDGE_INDS_TO_COLOR[pair], 3)
                               '''
        return img

    def verifyGameAngle(self, correctletter, marginError) :
        correctAngle = [0] * 2

        #[0] is left, [1] is right
        correctAngle[0] = LETTER_ANGLE[correctletter][0]
        correctAngle[1] = LETTER_ANGLE[correctletter][1]

        if self.records['x']['lshoulder'][-1] and self.records['x']['rshoulder'][-1] and self.records['x']['lshoulder'][-1] and self.records['x']['lelbow'][-1] and self.records['x']['relbow'][-1]:
            print("top body half detected")
            
            userLeftAngle  =  find_quad_angle(self.records['x']['lshoulder'][-1], self.records['y']['lshoulder'][-1],self.records['x']['lelbow'][-1],  self.records['y']['lelbow'][-1])
            #squatanglehip

            userRightAngle  =  find_quad_angle(self.records['x']['rshoulder'][-1], self.records['y']['rshoulder'][-1],self.records['x']['relbow'][-1],  self.records['y']['relbow'][-1])
            
            userAngle = [userLeftAngle, userRightAngle]
            print("left correct: ", correctAngle[0], "user: ", userLeftAngle,"|| right correct: ", correctAngle[1], "user: ", userRightAngle)
            if (
            userAngle[0] > correctAngle[0] - marginError and
            userAngle[0] < correctAngle[0] + marginError and
            userAngle[1] > correctAngle[1] - marginError and
            userAngle[1] < correctAngle[1] + marginError
            ) :
                return True
            else :
                return False
