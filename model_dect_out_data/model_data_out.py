import cv2
from backend.motion_detection.movenetidentifier import MovenetIdentifier
import numpy as np
import time
import tensorflow as tf
from model_dect_out_data.data_dump_and_read import Data_dump_part
from model_dect_out_data.sports_camera_mode import get_deg


class ModelDataOut:
    def __init__(self):
        self.lock_on_data_x = 0
        self.lock_on_data_y = 0
        self.model_out = 0
        self.video_handle = 0
        self.model_self = 0
        self.model_height = 0
        self.model_width = 0
        self.exercise_arg = 0
        self.img_height = 0
        self.img_width = 0
        self.listener = 0
        self.read_success = 0
        self.listener = 0
        self.out_video_handle = 0
        self.time_handle = 0
        self.time_handle_save = 0
        self.video_fps = 0
        self.model_composition = 0
        self.data = dict()
        self.video_handle_fps = 30
        self.video_s_from = ""
        self.video_read_start = False
        self.download_name = ''
        self.port_adress = 0
        self.is_camera = False
        self.obj_file = ""
        self.save_dump = ""

    def model_data_out_init(self, data_from, model_self_arg, download_name, port_adress, is_camera, obj_file):
        self.obj_file = obj_file
        self.save_dump = Data_dump_part(self.obj_file)
        self.is_camera = is_camera
        self.port_adress = port_adress
        self.download_name = download_name
        self.video_handle = cv2.VideoCapture(data_from)
        self.model_self = model_self_arg[0]
        self.model_height = model_self_arg[1]
        self.model_width = model_self_arg[2]
        self.model_composition = model_self_arg[3]
        self.data['x'] = dict()
        self.data['y'] = dict()
        for part in self.model_composition["detect_human_part_list"]:
            self.data['x'][part] = 0.0
            self.data['y'][part] = 0.0
        self.data['timing'] = 0.0

    def video_feed_stop(self):
        self.save_dump.save_data()
        self.time_handle_save = 0
        self.video_read_start = False
        self.video_handle.release()
        self.out_video_handle.release()
        return True

    def video_feed_start(self):

        self.read_success, img = self.video_handle.read()
        self.img_height, self.img_width, _ = img.shape
        self.video_handle_fps = int(self.video_handle.get(cv2.CAP_PROP_FPS))
        self.listener = MovenetIdentifier(self.img_height, self.img_width, self.exercise_arg)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out_video_handle = cv2.VideoWriter(self.download_name, fourcc, self.video_handle_fps,
                                                (self.img_width, self.img_height))
        print("start video feed")
        # self.out_video_sk_handle = cv2.VideoWriter(download_name_sk, fourcc, self.video_handle_fps,  (self.img_width, self.img_height))

    def get_video_all_info(self):
        width = int(self.video_handle.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.video_handle.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.video_handle.get(cv2.CAP_PROP_FPS))
        num_frames = int(self.video_handle.get(cv2.CAP_PROP_FRAME_COUNT))
        if num_frames > 9999999:
            num_frames = 99999
        video_info = {
            'width_i': width,
            'height_i': height,
            'fps_i': fps,
            'num_frames_i': num_frames,
            'from_i': self.video_s_from
        }
        return video_info

    def video_feed_source(self):
        if not self.video_read_start:
            self.video_feed_start()
            self.video_read_start = True
        self.time_handle = time.time()
        self.read_success, img = self.video_handle.read()
        if self.read_success:

            if self.is_camera:
                img = cv2.flip(img, 1)
                img = cv2.flip(img, -1)
            tf_img = cv2.resize(img, (self.model_width, self.model_height))
            tf_img = cv2.cvtColor(tf_img, cv2.COLOR_BGR2RGB)
            tf_img = np.asarray(tf_img)
            tf_img = np.expand_dims(tf_img, axis=0)

            # Resize and pad the image to keep the aspect ratio and fit the expected size.
            image = tf.cast(tf_img, dtype=tf.int32)

            # get np.array(17,3) to feed into listener
            args = self.model_self(image, self.img_height, self.img_width)
            # args = self.dtc.infer_keys_and_scores(image)
            # print(args)

            # update listener
            img = self.listener.update(args, img)

            time_taken = time.time() - self.time_handle
            self.time_handle_save += 1
            now_time = self.time_handle_save / self.video_handle_fps
            self.video_fps = float('{:.1f}'.format(1 / time_taken))
            self.time_handle = time.time()
            self.listener.records['time'].append(now_time)
            self.data['time'] = float(('{:.3f}'.format(self.listener.records['time'][-1])))

            for part in self.model_composition["detect_human_part_list"]:
                if self.listener.records['x'][part][-1]:
                    self.listener.records['x'][part][-1] = img.shape[0] - self.listener.records['x'][part][-1]
                    self.data['x'][part] = round(self.listener.records['x'][part][-1] / self.img_height * 100)
                #   data['x'][part][-1] /= 100
                if self.listener.records['y'][part][-1]:
                    self.listener.records['y'][part][-1] = img.shape[0] - self.listener.records['y'][part][-1]
                    self.data['y'][part] = round(self.listener.records['y'][part][-1] / self.img_width * 100)

                #  data['y'][part][-1] /= 100
                else:
                    continue

            output_frame = img.copy()
            self.out_video_handle.write(output_frame)
            (flag, encodedImage) = cv2.imencode(".jpg", output_frame)
            self.lock_on_data_x = self.data['x']['nose']
            self.lock_on_data_y = self.data['y']['nose']
            # print("x=" + str(self.lock_on_data_x))
            # print("y=" + str(self.lock_on_data_y))
            return_data = get_deg(data_trans=self.data)
            percent = self.save_dump.add_data(return_data)
            return self.read_success,  encodedImage, percent

        else:
            return self.read_success, None, None
