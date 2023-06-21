
import numpy as np
import math
import json
#normalize image________________________________________________________________ 
#Input signature: 1 int32 array
#Function: normalize array of image frame (impt for posenet)
#Output signature: float32
def normalize(img):
    img = np.asarray(img, dtype="float32")
    img = img / 127.5 - 1.0
    return img
#normalize image________________________________________________________________

#find_distance__________________________________________________________________
#Input signature: 4 floats x1, x2, y1, y2, where x1, y1 defines a point and x2, y2 defines another point
#Function: Find the Euclidean distance between the two points
#Output signature: float
def find_distance(x1, x2, y1, y2):
    distance = np.sqrt(np.square(x1-x2) + np.square(y1-y2))
    return distance
#find_distance__________________________________________________________________


#find_midpoint__________________________________________________________________
#Input signature: 4 floats x1, x2, y1, y2, where x1, y1 defines a point and x2, y2 defines another point
#Function: Find the Euclidean mid point between the two points
#Output signature: tuple of floats
def find_midpoint(x1, x2, y1, y2):
    midpoint = [(x1+x2)/2, (y1+y2)/2]
    return midpoint
#find_midpoint__________________________________________________________________

#find_angle_____________________________________________________________________
#Input signature: 3 floats a, b, c
#Function: Find angle
#Output signature: float in radians
def find_angle(a, b, c):
    # c^2 = a^2 + b^2 + 2abcostheta
    angle = np.arccos((np.square(c) - np.square(a) - np.square(b))/(2*a*b))
    return angle
#find_angle_____________________________________________________________________

#find_quad_angle_____________________________________________________________________
#Input signature: 4 floats y1, y2, x1, x2, where x1,y1 defines origin, x2,y2 defines end point
#Function: Find quadrantal angle, where zero degrees lies on positive x axis.
#Output signature: float in degrees
def find_quad_angle(x1,y1,x2,y2):
    angle = ((math.atan2(y2 - y1, x2-x1) * 180) / math.pi) * -1
    return angle
#find_quad_angle_____________________________________________________________________

#sigmoid________________________________________________________________________
#Input signature: confidence float
#Function: calc sigmoid
#Output signature: float
def sigmoid(base):
    return 1 / (1 + np.exp(-base))

#sigmoid________________________________________________________________________


#load_scores________________________________________________________________________
#Input signature: file name
#Function: load high score
#Output signature: float
def load_scores():
    with open("./data/scores.json") as infile:
        return json.load(infile)

#load_scores________________________________________________________________________

#save_scores________________________________________________________________________
#Input signature: high score float
#Function: save high score
#Output signature: float
def save_scores(scores):
    with open("./data/scores.json", "w") as outfile:
        json.dump(scores, outfile)

#save_scores________________________________________________________________________
