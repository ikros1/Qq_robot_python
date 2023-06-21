import os
import time
import cv2
from queue import Queue
import threading
from model_dect_out_data.sports_camera_mode import calc_difference, set_part, ret_pa
from model_dect_out_data.model_load import BoneDetectionModel
from model_dect_out_data.model_data_input import ModelDataInput
from model_dect_out_data.model_data_out import ModelDataOut

parts = ['nose', 'leye', 'reye', 'lear', 'rear', 'lshoulder',
         'rshoulder', 'lelbow', 'relbow', 'lwrist', 'rwrist', 'lhip', 'rhip', 'lknee',
         'rknee', 'lankle', 'rankle']
joints = ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_hip',
          'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']


class Container:

    def __init__(self):
        self.parseargs = ret_pa().parse_args()
        self.model_class = BoneDetectionModel()
        self.model_data_from = ModelDataInput()
        self.model_def = self.model_class.model_init(self.parseargs)
        self.model_out = list()
        self.thread_0 = 0
        self.thread_1 = 0
        self.thread_2 = 0
        self.thread_3 = 0
        self.video_feed_queue = []
        self.down_load_name_list = []
        self.down_load_dump_name_list = []
        for i in range(4):
            self.video_feed_queue.append(Queue())
        self.video_feed_queue_flag = False
        self.video_already_start = False
        self.video_download_path = ""
        self.video_download_can_start = [True, True, True, True]
        self.video_info_json = list()
        self.video_info_json_flag = True
        self.deg_new_package = [{}, {}, {}, {}]
        self.deg_package_loading_flag = False
        self.usr_upload_file_flag = False
        self.usr_upload_file_name = ""
        self.video_from_info = ""
        self.read_over_flag = False
        self.analysis_data_package_full_percent = [0, 0, 0, 0]

    def feed_start(self):
        if self.video_already_start:
            return
        else:
            self.video_already_start = True

        if self.usr_upload_file_flag:
            self.video_from_info = "用户上传"
            self.model_data_from.model_reset(self.usr_upload_file_name)
            self.model_out_reset()
            self.video_info_json_flag = True
            self.video_info_json.clear()
            for info in self.model_out:
                in_fu = info.get_video_all_info()
                in_fu['from_i'] = self.video_from_info
                self.video_info_json.append(in_fu)
            self.video_info_json_flag = False
        else:
            self.video_from_info = "本机默认摄像头"
            self.model_data_from.model_reset("camera")
            self.model_out_reset()
            self.video_info_json_flag = True
            self.video_info_json.clear()
            for info in self.model_out:
                in_fu = info.get_video_all_info()
                in_fu['from_i'] = self.video_from_info
                self.video_info_json.append(in_fu)
            self.video_info_json_flag = False
            self.video_info_json_flag = True

        self.video_download_path = os.path.abspath("downloaded_video_save_dir")
        self.down_load_name_list = self.down_load_name_get("f")
        self.down_load_dump_name_list = self.down_load_name_get("d")

        for i in range(len(self.model_out)):
            self.video_download_can_start[i] = False

        for model in self.model_out:
            model.video_feed_start()
        self.video_feed_queue_flag = True
        if len(self.model_out) > 0:
            if len(self.model_out) == 1:
                self.thread_0 = threading.Thread(target=self.feed_start_task_0)
                self.thread_0.start()
            if len(self.model_out) == 2:
                self.thread_0 = threading.Thread(target=self.feed_start_task_0)
                self.thread_0.start()
                self.thread_1 = threading.Thread(target=self.feed_start_task_1)
                self.thread_1.start()
        self.video_info_json_flag = False

    # 定义线程执行的任务
    def feed_start_task_0(self):
        while self.video_feed_queue_flag:
            success, encoded_image, load_percent = self.model_out[0].video_feed_source()
            if success:
                self.video_feed_queue[0].put(encoded_image)
                self.analysis_data_package_full_percent[0] = load_percent
            else:
                self.read_over_flag = True

    def feed_start_task_1(self):
        while self.video_feed_queue_flag:
            success, encoded_image, load_percent = self.model_out[1].video_feed_source()
            if success:
                self.video_feed_queue[1].put(encoded_image)
                self.analysis_data_package_full_percent[1] = load_percent
            else:
                self.read_over_flag = True

    def feed_start_task_2(self):
        while self.video_feed_queue_flag:
            success, encoded_image, load_percent = self.model_out[2].video_feed_source()
            if success:
                self.video_feed_queue[2].put(encoded_image)
                self.analysis_data_package_full_percent[2] = load_percent
            else:
                self.read_over_flag = True

    def feed_start_task_3(self):
        while self.video_feed_queue_flag:
            success, encoded_image, load_percent = self.model_out[3].video_feed_source()
            if success:
                self.video_feed_queue[3].put(encoded_image)
                self.analysis_data_package_full_percent[3] = load_percent
            else:
                self.read_over_flag = True

    def feed_stop(self):
        if not self.video_already_start:
            return
        else:
            self.video_already_start = False
        # self.thread_1.stop()
        # self.thread_2.stop()
        self.video_feed_queue_flag = False
        for i in range(len(self.model_out)):
            if self.model_out[i].video_feed_stop():
                self.video_download_can_start[i] = True
        self.analysis_data_package_full_percent = [0, 0, 0, 0]
        self.read_over_flag = False
        self.model_out[0].video_fps = 0
        self.model_out.clear()

    def down_load_name_get(self, name):
        name_list = list()
        if name == "f":
            for li in self.model_out:
                name_list.append(li.download_name)
        if name == "d":
            for li in self.model_out:
                name_list.append(li.obj_file)

        return name_list

    def model_out_reset(self):
        self.down_load_name_list = []
        self.down_load_dump_name_list = []
        self.model_out.clear()
        for i in range(len(self.video_feed_queue)):
            while not self.video_feed_queue[i].empty():
                self.video_feed_queue[i].get()
        self.analysis_data_package_full_percent = [0, 0, 0, 0]
        for i in range(len(self.model_data_from.video_source)):
            model_data_out_one = ModelDataOut()
            model_data_out_one.model_data_out_init(data_from=self.model_data_from.video_source[i],
                                                   model_self_arg=self.model_def,
                                                   download_name=self.model_data_from.video_download_name[i],
                                                   port_adress=self.model_data_from.video_send_port[i],
                                                   is_camera=self.model_data_from.is_camera[i],
                                                   obj_file=self.model_data_from.dump_file_name[i])
            self.model_out.append(model_data_out_one)

    def charts_four_part_data_get(self, index):
        if len(self.model_out) == 0:
            return False, 0
        else:
            out = self.model_out[index].save_dump.return_chart_data()

        return True, out

    def get_real_fps(self):
        if len(self.model_out) > 0:
            return str(self.model_out[0].video_fps)
        else:
            return str(0)

    def charts_under_part_data_get(self, json_d, index):
        if len(json_d) == 0:
            return False, 0
        else:
            list_o = self.model_out[index].save_dump.return_under_chart_data(json_d)
            return True, list_o

    def analysis_data_package_full_percent_get(self, index):
        if len(self.model_out) == 0:
            return 0
        else:

            return self.analysis_data_package_full_percent[index]

    def lock_on_data_get(self):
        data_list = list()
        data_list.clear()
        for lock_info in self.model_out:
            if lock_info.is_camera:
                x, sp = calc_difference(lock_info.lock_on_data_x, lock_info.lock_on_data_y)
                port = lock_info.port_adress
                data_list.append({"forw": x, "speed": sp, "port": port})

        return data_list

    def percent_get(self, index):
        percent_data = dict()
        percent_data["part_1"] = str(round(self.analysis_data_package_full_percent_get(index=index) * 100, 2)) + "%"
        percent_data["part_2"] = str(round(self.analysis_data_package_full_percent_get(index=index) * 100, 2)) + "%"
        percent_data["part_3"] = str(round(self.analysis_data_package_full_percent_get(index=index) * 100, 2)) + "%"
        percent_data["part_4"] = str(round(self.analysis_data_package_full_percent_get(index=index) * 100, 2)) + "%"
        percent_data["part_5"] = str(round(self.analysis_data_package_full_percent_get(index=index) * 100, 2)) + "%"
        # print(percent_data)
        return percent_data

    def video_feed(self):
        video_capture = cv2.VideoCapture("web_app/static/css/video/wait.mp4")
        success, frame = video_capture.read()
        if not success:
            # 如果读取失败，则重置播放位置
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success, frame = video_capture.read()
        (flag, encoded_image) = cv2.imencode(".jpg", frame)
        while True:
            while len(self.model_out) > 0:
                if self.video_feed_queue[0].qsize() > 0:
                    encoded_image = self.video_feed_queue[0].get()

                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                       bytearray(encoded_image) + b'\r\n')

            success, frame = video_capture.read()
            if not success:
                # 如果读取失败，则重置播放位置
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            (flag, encoded_image) = cv2.imencode(".jpg", frame)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                   bytearray(encoded_image) + b'\r\n')

    def video_feed_sk(self):
        video_capture = cv2.VideoCapture("web_app/static/css/video/wait_sk.mp4")
        success, frame = video_capture.read()
        if not success:
            # 如果读取失败，则重置播放位置
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success, frame = video_capture.read()
        (flag, encoded_image) = cv2.imencode(".jpg", frame)
        while True:
            while len(self.model_out) > 1:
                if self.video_feed_queue[1].qsize() > 0:
                    encoded_image = self.video_feed_queue[1].get()

                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                       bytearray(encoded_image) + b'\r\n')

            success, frame = video_capture.read()
            if not success:
                # 如果读取失败，则重置播放位置
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            (flag, encoded_image) = cv2.imencode(".jpg", frame)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                   bytearray(encoded_image) + b'\r\n')
