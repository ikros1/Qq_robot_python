import os
import threading
import time
import urllib
from queue import Queue

import openai
from dotenv import load_dotenv
from pydub import AudioSegment
from aicore import answer_loop, fitch_info, warm_core
from memory_data import memory
from shen_all import receive_data
from text_to_wav_interface import Core_tts_ika
from tool_kit import send_message_to_group, send_file_in_japanese_to_group
from wisper_to_text import voice_to_text

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def my_thread(q, arg1):
    results = warm_core(arg1)
    q.put(results)


class Robot:
    def __init__(self):
        self.t_k = Core_tts_ika()
        self.memory = memory()
        self.post_address = os.getenv("SERVER_PORT_ADRESS")
        self.qq_number = os.getenv("QQ_NUM")

    def receive_data(self, data):
        # 下面代码用于新建线程
        t = threading.Thread(target=receive_data, args=(data,))
        t2 = threading.Thread(target=self.divide_data, args=(data,))
        t2.start()
        t.start()

    def divide_data(self, data):
        temp_memory = data
        all_txt = []
        send_flag = False
        if temp_memory["post_type"] == "message" and temp_memory["message_type"] == "group":
            from_group = temp_memory["group_id"]
            from_person = temp_memory["user_id"]
            for txt in temp_memory["message"]:
                if txt["type"] == "at":
                    if txt["data"]["qq"] != " ":
                        all_txt.append(
                            {"role": "user",
                             "content": "人物" + str(from_person) + " @了人物 " + str(txt["data"]["qq"])})
                        if txt["data"]["qq"] == os.getenv("QQ_NUM"):
                            send_flag = True
                if txt["type"] == "reply":
                    all_txt.append({"role": "user", "content": "人物" + str(from_person) + " 回复了消息"})
                if txt["type"] == "text":
                    if txt["data"]["text"] != " ":
                        all_txt.append(
                            {"role": "user", "content": "人物" + str(from_person) + "说：" + str(txt["data"]["text"])})

                if txt["type"] == "record":
                    url = txt["data"]["url"]
                    # 获取当前时间戳
                    timestamp = int(time.time())
                    file_name = "file/voice/" + str(timestamp) + ".amr"
                    urllib.request.urlretrieve(url, file_name)
                    mp3_name = "file/voice/" + str(timestamp) + ".mp3"

                    # 将amr转换为mp3
                    sound = AudioSegment.from_file(file_name, format="amr")
                    sound.export(mp3_name, format="mp3")

                    # 删除amr文件
                    # os.remove(file_name)
                    vo_info = voice_to_text(mp3_name)

                    if vo_info != " ":
                        all_txt.append(
                            {"role": "user", "content": "人物" + str(from_person) + "说 ：" + str(vo_info)})

            self.store_message(from_group=from_group, temp_memory=all_txt)

    def store_message(self, from_group, temp_memory):
        send_flag = False
        if from_group not in self.memory.group_memory:
            self.memory.group_memory[from_group] = {}
            self.memory.init_ikaros_memory(from_group)
        self.memory.clear_memory(from_group)
        for info in temp_memory:
            if "风纪委员" in info["content"]:
                send_flag = True
                self.memory.group_memory[from_group]["assistant_memory"].append(info)
            self.memory.group_memory[from_group]["group_memory"].append(info)
        if send_flag:
            self.message_answer(from_group)

    def message_answer(self, from_group):
        ikaros_answer = ""
        q = Queue()
        t2 = threading.Thread(target=my_thread,
                              args=(q, self.memory.group_memory[from_group]))
        t2.start()
        warm_core_flag = False
        if os.getenv("CONNECT_TO_INTERNET") == "True":
            success, key_words, answer_prompts = answer_loop(self.memory.group_memory[from_group])
        else:
            success = True

        while not warm_core_flag:
            time.sleep(0.5)
            if not q.empty():
                warm_core_flag, ikaros_answer = q.get()
        if success:
            print("不需要联网查询")
            send_info = {'type': 'text',
                         'data': {'text': str(ikaros_answer)}}
            send_message_to_group(from_group, send_info)
            self.memory.group_memory[from_group]["group_memory"].append(
                {"role": "assistant", "content": str(ikaros_answer)})
            self.memory.group_memory[from_group]["assistant_memory"] = []
            t1 = threading.Thread(target=send_file_in_japanese_to_group, args=(from_group, ikaros_answer, self.t_k))
            t1.start()
        else:
            self.memory.group_memory[from_group]["group_memory"].append(
                {"role": "assistant", "content": str(ikaros_answer)})
            print("需要联网查询")
            send_info = {'type': 'text',
                         'data': {'text': "您的需求需要联网查询，耗时较长，稍后为您展示查询结果"}}
            send_message_to_group(from_group, send_info)
            self.memory.group_memory[from_group] = answer_prompts
            t2 = threading.Thread(target=fitch_info,
                                  args=(key_words, self.memory.group_memory[from_group], from_group, []))
            t2.start()
