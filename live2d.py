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

    def face_data_update(self):
        success = True
        lshoulder_x = self.data['x']['lshoulder']
        lshoulder_y = self.data['y']['lshoulder']
        rshoulder_x = self.data['x']['rshoulder']
        rshoulder_y = self.data['y']['rshoulder']
        nose_x = self.data['x']['nose']
        nose_y = self.data['y']['nose']
        if nose_x < 1:
            nose_x = 1
            success = False
        if nose_x > 99:
            nose_x = 99
            success = False
        if nose_y < 1:
            nose_y = 1
            success = False
        if nose_y > 99:
            nose_y = 99
            success = False
        if lshoulder_x < 1:
            lshoulder_x = 1
            success = False
        if lshoulder_x > 99:
            lshoulder_x = 99
            success = False
        if lshoulder_y < 1:
            lshoulder_y = 1
            success = False
        if lshoulder_y > 99:
            lshoulder_y = 99
            success = False
        if rshoulder_x < 1:
            rshoulder_x = 1
            success = False
        if rshoulder_x > 99:
            rshoulder_x = 99
            success = False
        if rshoulder_y < 1:
            rshoulder_y = 1
            success = False
        if rshoulder_y > 99:
            rshoulder_y = 99
            success = False

        # nose_x 取小数点后两位
        nose_x = int(nose_x) / 100
        nose_y = int(nose_y) / 100
        lshoulder_x = int(lshoulder_x) / 100
        lshoulder_y = int(lshoulder_y) / 100
        rshoulder_x = int(rshoulder_x) / 100
        rshoulder_y = int(rshoulder_y) / 100

        print("nose_x : ", nose_x, "nose_y : ", nose_y, "lshoulder_x : ", lshoulder_x, "lshoulder_y : ", lshoulder_y,
              "rshoulder_x : ", rshoulder_x, "rshoulder_y : ", rshoulder_y)

        return nose_x, nose_y, lshoulder_x, lshoulder_y, rshoulder_x, rshoulder_y, success

    def crop_frame(self, frame, x_start, x_end, y_start, y_end):
        height, width, _ = frame.shape
        x_start = int(width * x_start)
        x_end = int(width * x_end)
        y_start = int(height * y_start)
        y_end = int(height * y_end)
        cropped_frame = frame[y_start:y_end, x_start:x_end]
        return cropped_frame

    def rotate_and_crop(self, image, nose_x, nose_y, lshoulder_x, lshoulder_y, rshoulder_x, rshoulder_y):
        # 计算鼻子和肩膀的中心点坐标
        jj = image
        nose_center = (int(nose_x * image.shape[1]), int(nose_y * image.shape[0]))
        print("nose_center : ", nose_center)
        print("image.shape[1] : ", image.shape[1])
        print("image.shape[0] : ", image.shape[0])
        shoulder_center = (
            int((lshoulder_x + rshoulder_x) / 2 * image.shape[1]),
            int((lshoulder_y + rshoulder_y) / 2 * image.shape[0]))
        print("shoulder_center : ", shoulder_center)
        # 计算肩膀宽度
        shoulder_width = int(
            np.linalg.norm(np.array([lshoulder_x, lshoulder_y]) - np.array([rshoulder_x, rshoulder_y])) * image.shape[
                1])
        if shoulder_width < 0:
            shoulder_width = -shoulder_width
        # 计算旋转角度
        angle = np.degrees(np.arctan2(nose_center[1] - shoulder_center[1], nose_center[0] - shoulder_center[0]))
        angle = angle - 90
        print("angle : ", angle)
        # 计算鼻子中心点到肩膀的中心点的距离
        distance = np.linalg.norm(np.array(nose_center) - np.array(shoulder_center))
        # 执行旋转
        M = cv2.getRotationMatrix2D(nose_center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=(0, 0, 0))

        # 计算旋转后的坐标
        rotated_nose_center = np.dot(M, np.array([nose_center[0], nose_center[1], 1])).astype(int)
        rotated_shoulder_center = np.dot(M, np.array([shoulder_center[0], shoulder_center[1], 1])).astype(int)
        print("rotated_nose_center : ", rotated_nose_center)
        print("rotated_shoulder_center : ", rotated_shoulder_center)
        # 计算裁剪区域的边界
        """
        x1 = rotated_nose_center[0] - int(shoulder_width/2)
        x2 = rotated_nose_center[0] + int(shoulder_width/2)
        y1 = rotated_nose_center[1] - int(distance)
        y2 = rotated_nose_center[1] + int(distance/2)"""
        x1 = rotated_nose_center[0] - 100
        x2 = rotated_nose_center[0] + 100
        y1 = rotated_nose_center[1] - 100
        y2 = rotated_nose_center[1] + 100
        print("x1 : ", x1, "x2 : ", x2, "y1 : ", y1, "y2 : ", y2)

        # 裁剪图像
        cropped_image = rotated_image[y1:y2, x1:x2]
        img_D = self.crop_frame(jj, nose_y-0.2, nose_y+0.2, nose_x-0.2, nose_x+0.2)
        # return cropped_image
        return img_D

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
            nose_x, nose_y, lshoulder_x, lshoulder_y, rshoulder_x, rshoulder_y, success_d = self.face_data_update()
            if not success_d:
                continue
            # 截取图像
            face_image = self.rotate_and_crop(img, nose_x, nose_y, lshoulder_x, lshoulder_y, rshoulder_x, rshoulder_y)

            # 生成保存路径和文件名
            timestamp = int(time.time())
            save_path = os.path.join(save_folder, f"face_{timestamp}.jpg")

            # 显示图像
            if face_image.shape[0] > 0 and face_image.shape[1] > 0:  # 检查图像尺寸是否大于0
                cv2.imshow("face", face_image)
                cv2.imwrite(save_path, face_image)

            # 保存图像

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.video_handle.release()
        cv2.destroyAllWindows()
