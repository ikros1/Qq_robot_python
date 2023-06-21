import threading
import time
import os
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

    def face_data_update(self, img_width, img_height):
        # 检查是否检测到人脸
        if self.data['x']['nose'] == 0 and self.data['y']['nose'] == 0:
            # 返回整个图像范围
            x_min = 0
            y_min = 0
            x_max = img_width
            y_max = img_height
        else:
            # 获取肩膀和鼻子的坐标
            shoulder_x = self.data['x']['lshoulder'] + self.data['x']['rshoulder']
            shoulder_y = self.data['y']['lshoulder'] + self.data['y']['rshoulder']
            nose_x = self.data['x']['nose']
            nose_y = self.data['y']['nose']

            # 计算截取区域的坐标
            x_min = int(min(max(0, shoulder_x), max(0, nose_x)) / 100 * img_width)  # 左上角x坐标
            y_min = int(max(0, (nose_y - (shoulder_x - nose_x) * 2)) / 100 * img_height)  # 左上角y坐标
            x_max = int(max(min(img_width, shoulder_x), min(img_width, nose_x)) / 100 * img_width)  # 右下角x坐标
            y_max = int(min(img_height, nose_y) / 100 * img_height)  # 右下角y坐标

        # 输出坐标
        # print(f"({x_min}, {y_min}), ({x_max}, {y_max})")
        print("ll" + str(self.data['x']['lshoulder']))
        print("rr" + str(self.data['x']['rshoulder']))
        return y_min, x_min, y_max, x_max

    def face_look(self):
        while self.flag:
            time.sleep(0.2)
            img = self.video_handle.read()[1]

            img_height, img_width, _ = img.shape
            tf_img = cv2.resize(img, (self.model_width, self.model_height))
            tf_img = cv2.cvtColor(tf_img, cv2.COLOR_BGR2RGB)
            tf_img = np.asarray(tf_img)
            tf_img = np.expand_dims(tf_img, axis=0)

            # Resize and pad the image to keep the aspect ratio and fit the expected size.
            image = tf.cast(tf_img, dtype=tf.int32)

            # get np.array(17,3) to feed into listener
            args = self.model_self(image, img_height, img_width)
            img = self.listener.update(args, img)
            for part in self.model_composition["detect_human_part_list"]:
                if self.listener.records['x'][part][-1]:
                    self.listener.records['x'][part][-1] = img.shape[0] - self.listener.records['x'][part][-1]
                    self.data['x'][part] = round(self.listener.records['x'][part][-1] / img_height * 100)
                if self.listener.records['y'][part][-1]:
                    self.listener.records['y'][part][-1] = img.shape[0] - self.listener.records['y'][part][-1]
                    self.data['y'][part] = round(self.listener.records['y'][part][-1] / img_width * 100)

            self.json_str['x'] = self.data['x']['nose']
            self.json_str['y'] = self.data['y']['nose']

            # 创建保存图像的文件夹
            save_folder = 'C:\\Users\\ikaros\\Desktop\\Qq_robot_python\\face_id_dump'
            os.makedirs(save_folder, exist_ok=True)
            y_min, x_min, y_max, x_max = self.face_data_update(img_width, img_height)

            # 截取图像
            face_image = img[y_min:y_max, x_min:x_max]

            # 生成保存路径和文件名
            timestamp = int(time.time())
            save_path = os.path.join(save_folder, f"face_{timestamp}.jpg")

            # 显示图像
            cv2.imshow("face", face_image)

            # 保存图像
            cv2.imwrite(save_path, face_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.video_handle.release()
        cv2.destroyAllWindows()
