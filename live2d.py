import math
import time
import cv2
from backend.motion_detection.movenetidentifier import MovenetIdentifier
import numpy as np
import tensorflow as tf
from model_dect_out_data.model_load import BoneDetectionModel
from model_dect_out_data.sports_camera_mode import ret_pa


class FaceDetection:
    def __init__(self):
        self.video_handle = None
        self.flag = True
        self.listener = None
        parseargs = ret_pa().parse_args()
        model_class = BoneDetectionModel()
        model_def = model_class.model_init(parseargs)
        self.model_self = model_def[0]
        self.model_width = model_def[1]
        self.model_height = model_def[2]
        self.model_composition = model_def[3]

        self.data = dict()
        self.data['x'] = dict()
        self.data['y'] = dict()
        for part in self.model_composition["detect_human_part_list"]:
            self.data['x'][part] = 0.0
            self.data['y'][part] = 0.0
        self.data['timing'] = 0.0
        self.json_str = {"x": 0, "y": 0}

    def start(self):
        self.video_handle = cv2.VideoCapture(0)
        width = int(self.video_handle.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.video_handle.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.listener = MovenetIdentifier(height, width, 0)

    def return_data(self):
        return self.json_str

    def face_look(self):
        while self.flag:
            time.sleep(0.2)
            img = self.video_handle.read()[1]
            img = cv2.flip(img, 1)

            img_height, img_width, _ = img.shape
            tf_img = cv2.resize(img, (self.model_width, self.model_height))
            tf_img = cv2.cvtColor(tf_img, cv2.COLOR_BGR2RGB)
            tf_img = np.asarray(tf_img)
            tf_img = np.expand_dims(tf_img, axis=0)

            # Resize and pad the image to keep the aspect ratio and fit the expected size.
            image = tf.cast(tf_img, dtype=tf.int32)

            # get np.array(17,3) to feed into listener
            args = self.model_self(image, img_height, img_width)
            current = args
            curr_confidence = current[:, 2]
            curr_confidence = np.reshape(curr_confidence, (17, 1))

            x_prime = current[:, 1]
            x_prime = np.reshape(x_prime, (17, 1))

            y_prime = current[:, 0]
            y_prime = np.reshape(y_prime, (17, 1))
            index = 0
            for part in self.model_composition["detect_human_part_list"]:
                self.data['x'][part] = int(x_prime[index][0])
                self.data['y'][part] = int(y_prime[index][0])
                index += 1
            # print(self.data)
            # img = self.listener.update(args, img)
            my_img_height, my_img_width, _ = img.shape
            print("my_img_height : ", my_img_height)
            print("my_img_width : ", my_img_width)
            ear_width = (self.data['x']['lear'] - self.data['x']['rear']) * (
                    self.data['x']['lear'] - self.data['x']['rear']) + (
                                self.data['y']['lear'] - self.data['y']['rear']) * (
                                self.data['y']['lear'] - self.data['y']['rear'])
            ear_width = int(math.sqrt(ear_width) * 2)
            ear_width = int(ear_width / 2 + 10)
            angle = np.degrees(
                np.arctan2(self.data['x']['nose'] - (self.data['x']['leye'] + self.data['x']['reye']) / 2,
                           self.data['y']['nose'] - (self.data['y']['leye'] + self.data['y']['reye']) / 2))
            angle = -angle
            print("angle : ", angle)
            img = img[self.data['y']['nose'] - ear_width:self.data['y']['nose'] + ear_width,
                  self.data['x']['nose'] - ear_width:self.data['x']['nose'] + ear_width]
            nose_center = (ear_width, ear_width)
            if img.shape[0] > 0 and img.shape[1] > 0 and ear_width > 0:  # 检查图像尺寸是否大于0
                M = cv2.getRotationMatrix2D(nose_center, angle, 1.0)
                rotated_image = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_CONSTANT,
                                               borderValue=(0, 0, 0))
                img = rotated_image[
                      int(rotated_image.shape[1] / 2 - ear_width / 2):int(rotated_image.shape[1] / 2 + ear_width / 2),
                      int(rotated_image.shape[0] / 2 - ear_width / 2):int(rotated_image.shape[0] / 2 + ear_width / 2)]
                print("ear_width : ", ear_width)
                print("rotated_image.shape : ", img.shape)
                if img.shape[0] > 0 and img.shape[1] > 0:  # 检查图像尺寸是否大于0
                    cv2.imshow("face", img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
