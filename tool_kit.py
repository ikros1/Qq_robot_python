import os
import difflib
import random
import re
import time

import openai
import requests
import json
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


def chinese_to_jp(output):
    response2 = openai.Completion.create(
        engine="text-davinci-003",
        prompt='Translate the following text to Japanese: "' + output + '"',
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0,
    )

    output2 = response2.choices[0].text
    return str(output2)


def post_json(url, data):
    # 将数据转换为JSON格式
    json_data = json.dumps(data)

    # 设置请求头
    headers = {'Content-type': 'application/json'}

    # 发送POST请求
    response = requests.post(url, data=json_data, headers=headers)

    # 返回响应内容
    return response.text


def create_ruqun_data_array():
    json_array = []

    data_list = [
        "未注册用户点击链接进入注册 http://ikaros.love/zhuce.html\n",
        "已经注册用户点击链接进行登录 http://ikaros.love/denglu.html\n",
        "已经登录的用户点击链接进行答题 http://ikaros.love/datijieguo.php\n",
        "答题完成自动看到结果，没有刷出结果，可以点击链接 http://ikaros.love/datijieguo2.php\n",
        "结果截图保存并at胡来的螺丝\n"
    ]

    for data in data_list:
        json_obj = {"data": {"text": data}, "type": "text"}
        json_array.append(json_obj)

    return json_array


def send_message_to_group(id, data):
    url = os.getenv("SERVER_PORT_ADRESS") + "send_msg"
    group_data = {
        "group_id": id,
        "message": data
    }
    # (group_data)
    # print(url)
    post_json(url, group_data)


def send_record_to_group(from_group, file_path):
    print("send_record_to_group")

    url = os.getenv("SERVER_PORT_ADRESS") + "send_msg"
    group_data = {
        "group_id": from_group,
        "message": [{'type': 'record', 'data': {'file': file_path}}]
    }
    post_json(url, group_data)


def send_file_in_japanese_to_group(from_group, chinese_txt, tts_core):
    language = os.getenv("TTS_LANGUAGE")
    txt_str = ""
    if language == "日本語":
        txt_str = chinese_to_jp(chinese_txt)
        txt_str = txt_str.replace(" ", "").replace("\n", "")
    elif language == "简体中文":
        txt_str = chinese_txt.replace(" ", "").replace("\n", "")
    speaker = os.getenv("TTS_SPEAKER")
    speed = os.getenv("TTS_SPEED")
    # 将speed转换为float
    speed = float(speed)
    tts_front_path = os.getenv("TTS_FRONT_PATH")
    front_path = "data/voices/"
    # 通过时间戳赋予随机名字
    file_name = str(int(time.time())) + str(random.randint(0, 1000))
    file_back = ".wav"
    file_name_all = tts_front_path + front_path + file_name + file_back
    send_file_name_all = file_name + file_back
    # print(txt_str, speaker, language, speed, file_name_all)
    tts_core.tts_vo(text=txt_str, speaker=speaker, language=language, speed=speed, file_path=file_name_all)
    send_record_to_group(from_group=from_group, file_path=send_file_name_all)


def send_message_dati(group_id):
    send_message_to_group(group_id, create_ruqun_data_array())


def send_message_to_person(id, data):
    url = os.getenv("SERVER_PORT_ADRESS") + "send_msg"
    person_data = {
        "user_id": id,
        "message": data
    }
    # print(url)
    post_json(url, person_data)


def is_similar(str1, str2):
    seq = difflib.SequenceMatcher(None, str1, str2)
    similarity_ratio = seq.ratio()
    print("相似度：" + str(similarity_ratio))
    if similarity_ratio >= 0.7:
        return True
    else:
        return False


def load_json_str(text):
    json_pattern = re.compile(r'{.*?}')
    json_str = json_pattern.search(text).group()
    return json_str
