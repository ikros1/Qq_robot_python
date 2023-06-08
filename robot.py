import os
import threading
import time
import urllib
import openai
from dotenv import load_dotenv
from pydub import AudioSegment
from aicore import warm_core, AnswerLoop
from memory_data import memory
from shen_all import receive_data
from text_to_wav_interface import Core_tts_ika
from tool_kit import send_message_to_group, send_file_in_japanese_to_group
from wisper_to_text import voice_to_text

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


class Robot:
    def __init__(self):
        self.t_k = Core_tts_ika()
        self.memory = memory()
        self.post_address = os.getenv("SERVER_PORT_ADRESS")
        self.qq_number = os.getenv("QQ_NUM")
        self.web_answer_core = AnswerLoop()
        self.memory.admin_authority(str(os.getenv("MANAGER_QQ")))

    def receive_data(self, data):
        # 下面代码用于新建线程
        t = threading.Thread(target=receive_data, args=(data,))
        t2 = threading.Thread(target=self.divide_data, args=(data,))
        t2.start()
        t.start()

    def divide_data(self, data):
        temp_memory = data

        all_txt = []
        if temp_memory["post_type"] == "message" and temp_memory["message_type"] == "group":
            from_group = temp_memory["group_id"]
            if temp_memory["sender"]["user_id"]:
                from_person = str(temp_memory["sender"]["user_id"])
            else:
                from_person = "0000000000"
            for txt in temp_memory["message"]:
                if txt["type"] == "at":
                    if txt["data"]["qq"] != " ":
                        if txt["data"]["qq"] == os.getenv("QQ_NUM"):
                            all_txt.append({"from_person": from_person, "message": {"role": "user", "content": "人物" + str(from_person) + " @了人物风纪委员 "}})
                        else:
                            all_txt.append({"from_person": from_person, "message": {"role": "user", "content": "人物" + str(from_person) + " @了人物 " + str(txt["data"]["qq"])}})

                if txt["type"] == "reply":
                    all_txt.append({"from_person": from_person, "message": {"role": "user", "content": "人物" + str(from_person) + " 回复了消息"}})
                if txt["type"] == "text":
                    if txt["data"]["text"] != " ":
                        all_txt.append({"from_person": from_person, "message": {"role": "user", "content": "人物" + str(from_person) + "说：" + str(txt["data"]["text"])}})

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
                        all_txt.append({"from_person": from_person, "message": {"role": "user", "content": "人物" + str(from_person) + "说 ：" + str(vo_info)}})

            self.store_message(from_group=from_group, temp_memory=all_txt)

    def store_message(self, from_group, temp_memory):
        send_flag = False
        web_search_flag = False
        master_flag = False
        terminal = ""
        from_person = ""
        if from_group not in self.memory.group_memory:
            self.memory.group_memory[from_group] = {}
            self.memory.init_ikaros_memory(from_group)
        self.memory.clear_memory(from_group)
        for info in temp_memory:
            if not info["from_person"] in self.memory.authority:
                self.memory.init_authority(info["from_person"])
            if os.getenv("MANAGER_QQ") == info["from_person"] and "命令" in info["message"]["content"]:
                master_flag = True
                terminal = info["message"]["content"].replace("命令", "")
            if "风纪委员" in info["message"]["content"]:
                send_flag = True
                from_person = info["from_person"]
                self.memory.group_memory[from_group]["assistant_memory"].append(info["message"])
            if "联网查询" in info["message"]["content"]:
                web_search_flag = True
                from_person = info["from_person"]
            self.memory.group_memory[from_group]["group_memory"].append(info["message"])
        if send_flag:
            self.message_answer(from_group, web_search_flag, master_flag, terminal, from_person)

    def message_answer(self, from_group, web_search_flag, master_flag, terminal, from_person):
        if os.getenv("CONNECT_TO_INTERNET") == "True" and web_search_flag and self.memory.authority[from_person]["use_ai_web_search"]:
            def web_answer(group_memory, from_group_lan):
                self.web_answer_core.data_set(group_memory, from_group_lan)
                self.web_answer_core.run()

            t2 = threading.Thread(target=web_answer, args=(self.memory.group_memory[from_group], from_group))
            t2.start()

        else:
            if web_search_flag and os.getenv("CONNECT_TO_INTERNET") == "False":
                send_info = {'type': 'text', 'data': {'text': "您的需求需要联网查询，但是当前机器人关闭联网查询功能"}}
                send_message_to_group(from_group, send_info)
                return None
            if web_search_flag and not self.memory.authority[from_person]["use_ai_web_search"]:
                send_info = {'type': 'text', 'data': {'text': "您的需求需要联网查询，但您未获得联网查询权限"}}
                send_message_to_group(from_group, send_info)
                return None

            def my_thread(group_memory, from_group_lan):
                success, ikaros_answer = warm_core(group_memory)
                if success:
                    send_message_to_group(from_group_lan, {'type': 'text', 'data': {'text': str(ikaros_answer)}})
                    self.memory.group_memory[from_group_lan]["group_memory"].append({"role": "assistant", "content": str(ikaros_answer)})
                    if self.memory.authority[from_person]["use_ai_voice"]:
                        t1 = threading.Thread(target=send_file_in_japanese_to_group,
                                              args=(from_group_lan, ikaros_answer, self.t_k))
                        t1.start()

                else:
                    send_message_to_group(from_group_lan, {'type': 'text', 'data': {'text': "机器人连接openai出现了一些问题，请联系管理员"}})
            if self.memory.authority[from_person]["use_ai"]:
                t2 = threading.Thread(target=my_thread,
                                      args=(self.memory.group_memory[from_group], from_group))
                t2.start()
            else:
                send_message_to_group(from_group, {'type': 'text', 'data': {'text': str(from_person)+"您的权限不足无法使用AI"}})

