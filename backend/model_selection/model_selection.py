import numpy as np
import tensorflow as tf
import cv2
import time

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

class MovenetIdentifier:
    def __init__(self,height, width, seconds_per_slice = 0.05, threshold = 0.15, stickiness = 0.9, fugacity = 1.0):
        print("Calling init MI")
        self.width = width
        self.height = height
        self.seconds_per_slice = 0.05

        self.confidence = np.zeros((17,1))
        self.x = np.zeros((17,1))
        self.y = np.zeros((17,1))
        self.stickiness = stickiness
        self.fugacity = fugacity

        self.threshold = threshold

        self.opening_msg = False
        self.got_data = False


        self.jump_counter = 0
        self.idx_need = [0,1,2,5,6,13,14,15,16]
        self.idx_feet = [15,16]
        self.jump_memory = []

        self.memory = None

        self.state = 0 #0 = not jumping, 1 = jumping

        self.opening_msg = False
        self.got_data = False
        self.standing = np.zeros((17,1))
        self.squatting = np.zeros((17,1))


        self.squat_counter = 0
        self.idx_need = [0, 1, 2, 3,5, 6, 11, 12]

        self.start_in_seconds = 0


        self.times = []
        self.estimate = []
        self.target_high = []
        self.target_low = []

        '''

        self.graph_x = np.linspace(0,100,100)
        self.graph_y_head = np.zeros(100)
        self.graph_y_shoulder = np.zeros(100)
        self.graph_y_hip = np.zeros(100)
        self.graph_y_knee = np.zeros(100)
        self.graph_y_feet = np.zeros(100)
        #########################################

        self.graph_x_nose = np.zeros(100)
        self.graph_x_leye = np.zeros(100)
        self.graph_x_reye = np.zeros(100)
        self.graph_x_lear = np.zeros(100)
        self.graph_x_rear = np.zeros(100)
        self.graph_x_lshoulder = np.zeros(100)
        self.graph_x_rshoulder = np.zeros(100)
        self.graph_x_lelbow = np.zeros(100)
        self.graph_x_relbow = np.zeros(100)
        self.graph_x_lwrist = np.zeros(100)
        self.graph_x_rwrist = np.zeros(100)
        self.graph_x_lhip = np.zeros(100)
        self.graph_x_rhip = np.zeros(100)
        self.graph_x_lknee = np.zeros(100)
        self.graph_x_rknee = np.zeros(100)
        self.graph_x_lankle = np.zeros(100)
        self.graph_x_rankle = np.zeros(100)

        self.graph_y_nose = np.zeros(100)
        self.graph_y_leye = np.zeros(100)
        self.graph_y_reye = np.zeros(100)
        self.graph_y_lear = np.zeros(100)
        self.graph_y_rear = np.zeros(100)
        self.graph_y_lshoulder = np.zeros(100)
        self.graph_y_rshoulder = np.zeros(100)
        self.graph_y_lelbow = np.zeros(100)
        self.graph_y_relbow = np.zeros(100)
        self.graph_y_lwrist = np.zeros(100)
        self.graph_y_rwrist = np.zeros(100)
        self.graph_y_lhip = np.zeros(100)
        self.graph_y_rhip = np.zeros(100)
        self.graph_y_lknee = np.zeros(100)
        self.graph_y_rknee = np.zeros(100)
        self.graph_y_lankle = np.zeros(100)
        self.graph_y_rankle = np.zeros(100)
        '''
        ############################################
        '''
        plt.ion()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)

        self.line_head_y, = self.ax.plot(self.graph_x, self.graph_y_head, 'r-', label='head') # Returns a tuple of line objects, thus the comma
        self.line_shoulder_y, = self.ax.plot(self.graph_x, self.graph_y_shoulder, 'g-', label='shoulders')
        self.line_hip_y, = self.ax.plot(self.graph_x, self.graph_y_hip, 'b-', label='hips')
        self.line_knee_y, = self.ax.plot(self.graph_x, self.graph_y_knee, 'y-', label='knees')
        self.line_feet_y, = self.ax.plot(self.graph_x, self.graph_y_feet, 'k-', label='feet')

        ############################################

        self.line_x_nose, = self.ax.plot(self.graph_x, self.graph_x_nose, 'r-', label='x_nose')
        self.line_x_leye, = self.ax.plot(self.graph_x, self.graph_x_leye, 'g-', label='x_leye')
        self.line_x_reye, = self.ax.plot(self.graph_x, self.graph_x_reye, 'b-', label='x_reye')
        self.line_x_lear, = self.ax.plot(self.graph_x, self.graph_x_lear, 'y-', label='x_lear')
        self.line_x_rear, = self.ax.plot(self.graph_x, self.graph_x_rear, 'k-', label='x_rear')
        self.line_x_lshoulder, = self.ax.plot(self.graph_x, self.graph_x_lshoulder, 'r-', label='x_lshoulder')
        self.line_x_rshoulder, = self.ax.plot(self.graph_x, self.graph_x_rshoulder, 'g-', label='x_rshoulder')
        self.line_x_lelbow, = self.ax.plot(self.graph_x, self.graph_x_lelbow, 'b-', label='x_lelbow')
        self.line_x_relbow, = self.ax.plot(self.graph_x, self.graph_x_relbow, 'y-', label='x_relbow')
        self.line_x_lwrist, = self.ax.plot(self.graph_x, self.graph_x_lwrist, 'k-', label='x_lwrist')
        self.line_x_rwrist, = self.ax.plot(self.graph_x, self.graph_x_rwrist, 'r-', label='x_rwrist')
        self.line_x_lhip, = self.ax.plot(self.graph_x, self.graph_x_lhip, 'g-', label='x_lhip')
        self.line_x_rhip, = self.ax.plot(self.graph_x, self.graph_x_rhip, 'b-', label='x_rhip')
        self.line_x_lknee, = self.ax.plot(self.graph_x, self.graph_x_lknee, 'y-', label='x_lknee')
        self.line_x_rknee, = self.ax.plot(self.graph_x, self.graph_x_rknee, 'k', label='x_rknee')
        self.line_x_lankle, = self.ax.plot(self.graph_x, self.graph_x_lankle, 'r-', label='x_lankle')
        self.line_x_rankle, = self.ax.plot(self.graph_x, self.graph_x_rankle, 'r-', label='x_rankle')

        self.line_y_nose, = self.ax.plot(self.graph_x, self.graph_y_nose, 'r--', label='y_nose')
        self.line_y_leye, = self.ax.plot(self.graph_x, self.graph_y_leye, 'r--', label='y_leye')
        self.line_y_reye, = self.ax.plot(self.graph_x, self.graph_y_reye, 'r--', label='y_reye')
        self.line_y_lear, = self.ax.plot(self.graph_x, self.graph_y_lear, 'r--', label='y_lear')
        self.line_y_rear, = self.ax.plot(self.graph_x, self.graph_y_rear, 'r--', label='y_rear')
        self.line_y_lshoulder, = self.ax.plot(self.graph_x, self.graph_y_lshoulder, 'r--', label='y_lshoulder')
        self.line_y_rshoulder, = self.ax.plot(self.graph_x, self.graph_y_rshoulder, 'r--', label='y_rshoulder')
        self.line_y_lelbow, = self.ax.plot(self.graph_x, self.graph_y_lelbow, 'r--', label='y_lelbow')
        self.line_y_relbow, = self.ax.plot(self.graph_x, self.graph_y_relbow, 'r--', label='y_relbow')
        self.line_y_lwrist, = self.ax.plot(self.graph_x, self.graph_y_lwrist, 'r--', label='y_lwrist')
        self.line_y_rwrist, = self.ax.plot(self.graph_x, self.graph_y_rwrist, 'r--', label='y_rwrist')
        self.line_y_lhip, = self.ax.plot(self.graph_x, self.graph_y_lhip, 'r--', label='y_lhip')
        self.line_y_rhip, = self.ax.plot(self.graph_x, self.graph_y_rhip, 'r--', label='y_rhip')
        self.line_y_lknee, = self.ax.plot(self.graph_x, self.graph_y_lknee, 'r--', label='y_lknee')
        self.line_y_rknee, = self.ax.plot(self.graph_x, self.graph_y_rknee, 'r--', label='y_rknee')
        self.line_y_lankle, = self.ax.plot(self.graph_x, self.graph_y_lankle, 'r--', label='y_lankle')
        self.line_y_rankle, = self.ax.plot(self.graph_x, self.graph_y_rankle, 'r--', label='y_rankle')


        self.ax.legend()
        plt.xlim([0, 100])
        plt.ylim([0, height])
        '''

        self.records = dict()
        self.records['x'] = dict()
        self.records['y'] = dict()


        for part in parts:
            self.records['x'][part] = []
            self.records['y'][part] = []

        self.records['timing'] = []
        self.records['hip_angle'] = []
        self.records['knee_angle'] = []

        '''

        records : {
            x: {
                nose: []
                leye: []
                ..
                timing: []
            }
            y: {
                nose: []
                leye: []
                ..
            }
        }
        '''

        #################

    def update(self, current, img, img_sk):
        print("Calling update MI")
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
        self.confidence = (self.confidence ** 2 * self.stickiness + curr_confidence ** 2 * self.fugacity) / (self.confidence * self.stickiness+ curr_confidence * self.fugacity)


        for val_idx in range(17):
            if self.confidence[val_idx] > self.threshold:
                img = cv2.circle(img, (int(self.x[val_idx]),  int(self.y[val_idx])), 2, (0, 255, 0), 5)
                img_sk = cv2.circle(img_sk, (int(self.x[val_idx]),  int(self.y[val_idx])), 2, (0, 255, 0), 5)


        img, img_sk = self.draw_lines(img,img_sk)

        x_tmp = 0
        y_tmp = 0


        for part in parts:
            if self.confidence[parts.index(part)] > self.threshold:
                self.records['x'][part].append(self.x[parts.index(part)][0])
                self.records['y'][part].append(self.y[parts.index(part)][0])
            else:
                self.records['x'][part].append(None)
                self.records['y'][part].append(None)

        if self.got_data:
            squat_status = self.check_for_squat()
            if squat_status == 1:
                self.squat_counter += 1
                print("That's ", self.squat_counter, "!")
        else:
            self.get_basic_info()

        '''
        self.records['x']['nose'].append(self.x[0][0])S
        sel

        records: {
            x: {
                nose: [],
                ..
            }
            y: {
                nose: [],
                ..
            }
        }
        '''

        # stuff below can delete

        return img, img_sk

    def draw_lines(self, img, img_sk): #no change
        #img = cv2.line(img, (0,height//2), (width,height//2), (255,0,255), 3)
        print("Calling draw MI")
        for pair in KEYPOINT_EDGE_INDS_TO_COLOR:
            start,end = pair
            if self.confidence[start]> self.threshold and self.confidence[end]> self.threshold:
                img = cv2.line(img, (int(self.x[start]),int(self.y[start]))
                               , (int(self.x[end]),int(self.y[end]))
                               , KEYPOINT_EDGE_INDS_TO_COLOR[pair], 3)

                img_sk = cv2.line(img_sk, (int(self.x[start]),int(self.y[start]))
                               , (int(self.x[end]),int(self.y[end]))
                               , KEYPOINT_EDGE_INDS_TO_COLOR[pair], 3)
        return img, img_sk

    def get_basic_info(self):
        print("Calling basic MI")
        if not self.opening_msg:
            print("Please stand straight in front of the camera!")
            self.opening_msg = True

        #need nose 0, left eye 1, right eye 2, left shld 5, right shld 6, left hip 11, right hip 12, left knee 13, right knee 14



        if np.all(self.confidence[self.idx_need]> self.threshold):
            self.standing = self.y
            self.got_data = True

            #approximate height to lower shoulder be approx hips to knees
            self.length_approx = (self.y[13] + self.y[14] - self.y[11] - self.y[12]) * 0.9 // 2
            print(self.length_approx)
            self.squat_range = (self.length_approx ** 2 // 225)
            self.squat_range = 15
            self.start_in_seconds = time.time()

            self.squatting = self.standing + self.length_approx
            print("Thank you!, error range is: ", self.squat_range)
            return True
        return False


    def check_for_squat(self):
        print("Calling squat checker MI")
        #check if we are confident enough about where we are
        if np.any(self.confidence[self.idx_need]< self.threshold):
            return 0
        checker = []



        if self.state == 1:
            targets = self.squatting

            for i in self.idx_need:
                #if we are going down and a point is too far from the threshold
                if self.y[i] < targets[i] and (self.y[i] - targets[i]) ** 2 > self.squat_range:
                    return 0

        else:
            targets = self.standing


            for i in self.idx_need:
                #if we are going up and a point is too far from the threshold
                if self.y[i] > targets[i] and (self.y[i] - targets[i]) ** 2 > self.squat_range:
                    return 0

        self.state *= -1
        return self.state


    def add_graphics(self,flipped_img):
        print("Calling add graphics MI")
        total = 0
        count = 0

        for i in self.idx_need:
            if self.confidence[i] < self.threshold:
                continue
            total += self.squatting[i] - self.y[i]
            count += 1
        if count == 0 or not self.got_data:
            ratio = 0
        else:
            approx_dist_from_squatting = total/count
            ratio = approx_dist_from_squatting / self.length_approx

            if ratio < 0:
                ratio = 0
            elif ratio > 1:
                ratio = 1

        if self.got_data and len(self.times) < 30:
            self.times.append(time.time())
            self.estimate.append(ratio)
            self.target_high.append(1)
            self.target_low.append(0)


        bar_top = int(0.1*self.height)
        bar_bottom = bar_top * 7

        top_words = int(0.05 * self.height)

        bar_progress = (bar_bottom - bar_top) * ratio + bar_top


        bar_left = int(self.width * 0.9)
        bar_right = bar_left + 25
        bar_center = bar_left+10
        bot_words = int(0.8 * self.height)

        cv2.putText(flipped_img, f'{self.squat_counter}', (bar_center, top_words), cv2.FONT_HERSHEY_PLAIN, 2,
          (0, 255, 0), 2)

        cv2.rectangle(flipped_img, (bar_left, bar_top), (bar_right, bar_bottom), (0, 255, 0), 3)
        cv2.rectangle(flipped_img, (bar_left, int(bar_progress)), (bar_right, bar_bottom), (255, 0, 0), cv2.FILLED)




        cv2.putText(flipped_img, f'{int(ratio*100)}%', (bar_center, bot_words), cv2.FONT_HERSHEY_PLAIN, 2,
                  (0, 255, 0), 2)

        return flipped_img
