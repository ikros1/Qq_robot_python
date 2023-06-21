import cv2
import dlib
import numpy as np


class FaceDetection:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        # 加载人脸识别模型
        self.face_recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

    def find_face(self, img):
        known_faces = []
        known_ids = []
        known_descriptors = []  # 存储已知人脸的特征描述符

        # 添加已知人脸的图像和ID
        known_faces.append(cv2.imread("face_id_img/known_face_1.jpg"))
        known_ids.append(1)
        known_faces.append(cv2.imread("face_id_img/known_face_2.png"))
        known_ids.append(2)

        # 提取已知人脸的特征描述符
        for known_face in known_faces:
            # 转换为灰度图像
            gray = cv2.cvtColor(known_face, cv2.COLOR_BGR2GRAY)
            # 检测人脸
            face = self.detector(gray)[0]
            # 获取人脸关键点
            landmarks_known = self.predictor(gray, face)
            # 提取人脸特征
            known_descriptor = self.face_recognizer.compute_face_descriptor(known_face, landmarks_known)
            known_descriptors.append(known_descriptor)

        # 打开摄像头
        cap = cv2.VideoCapture(0)

        while True:
            # 读取摄像头帧
            ret, frame = cap.read()

            # 转换为灰度图像
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 检测人脸
            faces = self.detector(gray)

            # 对每个检测到的人脸进行处理
            for face in faces:
                # 获取人脸关键点
                landmarks = self.predictor(gray, face)

                # 提取人脸特征
                face_descriptor = self.face_recognizer.compute_face_descriptor(frame, landmarks)

                # 在已知人脸中匹配
                for i, known_descriptor in enumerate(known_descriptors):
                    # 计算特征之间的欧氏距离
                    distance = np.linalg.norm(np.array(face_descriptor) - np.array(known_descriptor))

                    # 如果距离小于阈值，则匹配成功
                    print(distance)
                    if distance < 0.3:
                        # 绘制人脸框和ID
                        x, y, w, h = face.left(), face.top(), face.width(), face.height()
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frame, str(known_ids[i]), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0),
                                    2)

            # 显示结果帧
            cv2.imshow("Face Recognition", frame)

            # 按下ESC键退出
            if cv2.waitKey(1) == 27:
                break

        # 释放资源
        cap.release()
        cv2.destroyAllWindows()
