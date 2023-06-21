import datetime
from dotenv import load_dotenv
import os

load_dotenv()


class ModelDataInput:
    def __init__(self):
        self.from_data = "camera"
        self.video_source = list()
        self.video_download_name = list()
        self.video_send_port = list()
        self.is_camera = list()
        self.dump_file_name = list()

    def model_reset(self, data):
        self.from_data = data
        self.video_source.clear()
        self.video_send_port.clear()
        self.video_download_name.clear()
        self.is_camera.clear()
        self.dump_file_name.clear()
        if self.from_data == "camera":

            self.video_source.append(os.getenv("camera_address_1"))
            self.video_source.append(os.getenv("camera_address_2"))
            date_string = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            self.video_download_name.append("downloaded_video_save_dir/output_1{}.avi".format(date_string))
            self.video_download_name.append("downloaded_video_save_dir/output_2{}.avi".format(date_string))
            self.dump_file_name.append("object_save/obj_1_{}.dump".format(date_string))
            self.dump_file_name.append("object_save/obj_2_{}.dump".format(date_string))
            self.video_send_port.append(os.getenv("camera_ptz_port_1"))
            self.video_send_port.append(os.getenv("camera_ptz_port_2"))
            if os.getenv("camera_is_flip_1") == "True":
                self.is_camera.append(True)
            else:
                self.is_camera.append(False)
            if os.getenv("camera_is_flip_2") == "True":
                self.is_camera.append(True)
            else:
                self.is_camera.append(False)

        else:
            video_name = self.from_data
            exercise_type = 'usr_upload_dir'
            self.video_source.append(exercise_type + '/' + video_name)
            date_string = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            self.video_download_name.append("downloaded_video_save_dir/output1_{}.avi".format(date_string))
            self.video_send_port.append(18911)
            self.is_camera.append(False)
            self.dump_file_name.append("object_save/obj_1_{}.dump".format(date_string))
