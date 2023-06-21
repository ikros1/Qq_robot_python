import pickle
import threading
import time

import numpy as np

from model_dect_out_data.sports_camera_mode import set_part


def smooth_data(data):
    smooth_kernel = 5
    kernel = np.ones(smooth_kernel) / smooth_kernel
    pad_width = (smooth_kernel - 1) // 2
    return_data = np.pad(np.convolve(data, kernel, 'valid'), pad_width, mode='edge').tolist()
    # 将return_data中的数据转换为float类型，保留小数点后两位
    return_data = [round(float(i), 2) for i in return_data]
    return return_data


class Data_dump_part:
    def __init__(self, file_handle):
        self.file_handle = file_handle
        # 构建一个字典,用于存储平滑后的数据
        self.smooth_data = None

        # 构建一个字典,用于存储数据
        self.temp_data = {"time": [], 'left_shoulder': [], 'right_shoulder': [], 'left_elbow': [], 'right_elbow': [],
                          'left_hip': [], 'right_hip': [], 'left_knee': [], 'right_knee': [], 'left_ankle': [],
                          'right_ankle': []}
        # 新建一个变量用于计算self.temp_data中“time”键对应的列表的长度
        self.temp_data_len = 0
        # 用于计数的变量index
        self.index = 0
        # 数据加载完成标志位
        self.read_success = False
        # 新建一个线程
        self.thread = None
        self.data_already = False
        self.data_had_read = False

    # 构建方法利用for循环将和self.temp_data同样的结构的字典传入,并将其追加到self.temp_data中，并且将self.temp_data_len加一
    def add_data(self, data):
        for key in self.temp_data:
            self.temp_data[key].append(data[key][0])
        self.temp_data_len += 1
        self.index += 1
        # 当self.index大于500时,调用smooth_temp_data方法
        if self.index > 500:
            # 新建线程,将self.smooth_temp_data方法传入，在运行时不会阻塞主线程，运行结束后会自动结束
            self.thread = threading.Thread(target=self.smooth_temp_data)
            self.thread.start()
            self.index = 0
            return 1
        else:
            return self.get_data_fill()

    # 构建方法用于返回数据填充百分比，保留小数点后两位
    def get_data_fill(self):
        return round(self.index / 500, 4)

    # 构建方法用于加载数据,将self.file_handle传入read_obj方法中,并将返回值赋值给self.temp_data
    def load_data(self):
        self.temp_data = self.read_obj()
        self.temp_data_len = len(self.temp_data["time"])
        self.read_success = True

    # 构建方法用于返回self.smooth_data
    def get_smooth_data(self):
        return self.smooth_data

    def return_chart_data(self):
        while not self.read_success:
            time.sleep(0.1)
        out_smooth = dict()
        out_smooth["pat_0"] = set_part(self.smooth_data["time"], self.smooth_data["left_elbow"],
                                       self.smooth_data["right_elbow"])
        out_smooth["pat_1"] = set_part(self.smooth_data["time"], self.smooth_data["left_shoulder"],
                                       self.smooth_data["right_shoulder"])
        out_smooth["pat_2"] = set_part(self.smooth_data["time"], self.smooth_data["left_hip"],
                                       self.smooth_data["right_hip"])
        out_smooth["pat_3"] = set_part(self.smooth_data["time"], self.smooth_data["left_knee"],
                                       self.smooth_data["right_knee"])
        return out_smooth

    def return_under_chart_data(self, json_d):
        while not self.read_success:
            time.sleep(0.1)
        list_o = [["Jon_name", "Deg", "Time"]]
        for jon in json_d:
            for i in range(len(self.smooth_data["time"])):
                list_o.append([jon, self.smooth_data[jon][i], self.smooth_data["time"][i]])

        return list_o

    # 构建方法用于平滑数据,将下标从self.temp_data_len-1到self.temp_data_len-500的数据传入smooth_data方法中,并将返回值赋值给self.smooth_data
    def smooth_temp_data(self):
        self.smooth_data = dict()
        self.read_success = False
        for key in self.temp_data:
            self.smooth_data[key] = smooth_data(self.temp_data[key][self.temp_data_len - 500:self.temp_data_len])
        self.read_success = True
        self.data_already = True
        self.data_had_read = False

    def clear_data(self):
        self.temp_data.clear()

    def save_data(self):
        self.write_boj(self.temp_data)

    def write_boj(self, obj):
        with open(self.file_handle, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL, fix_imports=True)

    def read_obj(self):
        with open(self.file_handle, 'rb') as f:
            loaded_list = pickle.load(f)

        return loaded_list
