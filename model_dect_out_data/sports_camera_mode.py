import argparse
import datetime
import json
import math
import socket
import time
from dotenv import load_dotenv
import os

load_dotenv()


import numpy as np
import psutil

joints = ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_hip',
          'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']


def get_deg(data_trans):
    org = dict()
    org["time"] = []
    for joint in joints:
        org[joint] = []
    if data_trans["time"]:
        org["time"].append(data_trans["time"])

        data_temp = angle_of_triangle(data_trans["x"]['lshoulder'], data_trans["y"]['lshoulder'],
                                      data_trans["x"]['lelbow'], data_trans["y"]['lelbow'],
                                      data_trans["x"]['lhip'], data_trans["y"]['lhip'])
        org['left_shoulder'].append(data_temp)

        data_temp = angle_of_triangle(data_trans["x"]['rshoulder'], data_trans["y"]['rshoulder'],
                                      data_trans["x"]['relbow'], data_trans["y"]['relbow'],
                                      data_trans["x"]['rhip'], data_trans["y"]['rhip'])
        org['right_shoulder'].append(data_temp)
        data_temp = angle_of_triangle(data_trans["x"]['lelbow'], data_trans["y"]['lelbow'],
                                      data_trans["x"]['lshoulder'], data_trans["y"]['lshoulder'],
                                      data_trans["x"]['lwrist'], data_trans["y"]['lwrist'])
        org['left_elbow'].append(data_temp)

        data_temp = angle_of_triangle(data_trans["x"]['relbow'], data_trans["y"]['relbow'],
                                      data_trans["x"]['rshoulder'], data_trans["y"]['rshoulder'],
                                      data_trans["x"]['rwrist'], data_trans["y"]['rwrist'])
        org['right_elbow'].append(data_temp)

        data_temp = angle_of_triangle(data_trans["x"]['lhip'], data_trans["y"]['lhip'],
                                      data_trans["x"]['lshoulder'], data_trans["y"]['lshoulder'],
                                      data_trans["x"]['lknee'], data_trans["y"]['lknee'])
        org['left_hip'].append(data_temp)

        data_temp = angle_of_triangle(data_trans["x"]['rhip'], data_trans["y"]['rhip'],
                                      data_trans["x"]['rshoulder'], data_trans["y"]['rshoulder'],
                                      data_trans["x"]['rknee'], data_trans["y"]['rknee'])
        org['right_hip'].append(data_temp)

        data_temp = angle_of_triangle(data_trans["x"]['lknee'], data_trans["y"]['lknee'],
                                      data_trans["x"]['lankle'], data_trans["y"]['lankle'],
                                      data_trans["x"]['lhip'], data_trans["y"]['lhip'])
        org['left_knee'].append(data_temp)

        data_temp = angle_of_triangle(data_trans["x"]['rknee'], data_trans["y"]['rknee'],
                                      data_trans["x"]['rankle'], data_trans["y"]['rankle'],
                                      data_trans["x"]['rhip'], data_trans["y"]['rhip'])
        org['right_knee'].append(data_temp)

        org['left_ankle'].append(0)

        org['right_ankle'].append(0)
    """

    smooth_org['left_shoulder'] = list(smooth_data(org['left_shoulder']))
    smooth_org['right_shoulder'] = list(smooth_data(org['right_shoulder']))
    smooth_org['left_elbow'] = list(smooth_data(org['left_elbow']))
    smooth_org['right_elbow'] = list(smooth_data(org['right_elbow']))
    smooth_org['left_hip'] = list(smooth_data(org['left_hip']))
    smooth_org['right_hip'] = list(smooth_data(org['right_hip']))
    smooth_org['left_knee'] = list(smooth_data(org['left_knee']))
    smooth_org['right_knee'] = list(smooth_data(org['right_knee']))
    """

    return org


def gauge_data_get():
    psutil.cpu_percent()
    time.sleep(0.1)
    psutil.cpu_percent()
    time.sleep(0.1)
    psutil.cpu_percent()
    time.sleep(0.1)
    cpu_percent = psutil.cpu_percent()
    return cpu_percent


def date_get():
    json_d = dict()
    json_d["date_t"] = datetime.datetime.now().strftime("%H:%M")
    json_d["date_y"] = datetime.datetime.now().strftime("%Y-%m-%d")
    weekdays = ['星期一', '星期二', '星期三', '星期四', '星期五', '星期六', '星期日']
    today = datetime.datetime.now().weekday()
    json_d["date_d"] = weekdays[today]
    return json_d




def ret_pa():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="type of model eg movenet, posenet", default="movenet", type=str)
    parser.add_argument(
        "--submodel", help="submodel", default="lightning", type=str
    )
    parser.add_argument(
        "--istf_lite", help="if movenet, 1 for lite, 0 for normal", default=1, type=int
    )
    parser.add_argument(
        "--tf_quant", help="if movenet and tflite, type of quantization", default="8int", type=str
    )
    parser.add_argument(
        "--load_type", help="is the model loaded local or nonlocal", default="local", type=str
    )
    parser.add_argument(
        "--data_from", help="where is data come from", default="camera", type=str
    )
    return parser


def is_triangle(a, b, c):
    if a + b > c and a + c > b and b + c > a:
        return True
    else:
        return False


def angle_of_triangle(x1, y1, x2, y2, x3, y3):
    a = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    b = math.sqrt((x1 - x3) ** 2 + (y1 - y3) ** 2)
    c = math.sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2)
    if is_triangle(a, b, c):
        out_d = math.degrees(math.acos((c ** 2 - a ** 2 - b ** 2) / (-2 * a * b)))
    else:
        out_d = 180

    if out_d is None:
        return 180
    else:
        return float('{:.2f}'.format(out_d))


def set_part(pac, x, y):
    out_p = {
        "xAxis": {
            "type": 'category',
            "data": pac
        },
        "yAxis": {
            "type": 'value'
        },
        "series": [
            {
                "data": x,
                "type": 'line',
                "smooth": "true"
            },
            {
                "data": y,
                "type": 'line',
                "smooth": "true"
            }

        ]
    }

    return dict(out_p)


def calc_difference(x, y):
    if x < 0:
        x = 0
    if x > 100:
        x = 100
    if y < 0:
        y = 0
    if y > 100:
        y = 100

    diff_x = abs(x - 50)
    diff_y = abs(y - 75)
    if diff_y < 5:
        diff_level_y = 0
    elif diff_y < 10:
        diff_level_y = 1
    elif diff_y < 15:
        diff_level_y = 2
    elif diff_y < 20:
        diff_level_y = 3
    elif diff_y < 25:
        diff_level_y = 4
    else:
        diff_level_y = 4

    if diff_x < 10:
        diff_level_x = 0
    elif diff_x < 20:
        diff_level_x = 1
    elif diff_x < 30:
        diff_level_x = 2
    elif diff_x < 40:
        diff_level_x = 3
    elif diff_x < 50:
        diff_level_x = 4
    else:
        diff_level_x = 5

    if 70 <= y <= 80:
        fwd_yy = 99
        speed_yy = 0
    elif y > 80:
        fwd_yy = 1
        speed_yy = diff_level_y
    else:
        fwd_yy = 0
        speed_yy = diff_level_y

    if 40 <= x <= 60:
        fwd_xx = 99
        speed_xx = 0

    elif x > 60:
        fwd_xx = 2
        speed_xx = diff_level_x
    else:
        fwd_xx = 3
        speed_xx = diff_level_x
    # print(x,y)
    if diff_x > diff_y:
        fwd = fwd_xx
        speed = speed_xx
    else:
        fwd = fwd_yy
        speed = speed_yy
    return fwd, speed


def send_data(lock_on_list):
    adress = os.getenv("camera_ptz_address")
    send_json_data(adress, lock_on_list)


def send_json_data(host, data_dict):
    """
    发送JSON数据到指定主机和端口

    :param host: 目标主机名或IP地址
    :param port: 目标端口号
    :param data_dict: 包含JSON数据的Python字典
    """

    # 创建一个socket对象
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port = int(data_dict["port"])

    # 连接到指定主机和端口号
    s.connect((host, port))

    # 把字典数据转换为JSON字符串
    message = json.dumps(data_dict)

    # 发送消息
    s.sendall(message.encode())

    # 关闭socket连接
    s.close()
