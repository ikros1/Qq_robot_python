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
from role_set import role_dict
from tool_kit import send_message_dati

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


class Robot:
    def __init__(self):
        self.robot_name_init = os.getenv("DEFAULT_ROLE")
        self.robot_voice_init = os.getenv("TTS_SPEAKER")
        self.robot_language_init = os.getenv("TTS_LANGUAGE")
        self.robot_speed_init = os.getenv("TTS_SPEED")
        self.robot_name = {}
        self.robot_voice = {}
        self.robot_language = {}
        self.robot_speed = {}
        self.t_k = Core_tts_ika()
        self.memory = memory()
        self.robot_language_list = ["日本語", "简体中文"]
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
            if from_group not in self.robot_name:
                self.robot_name[from_group] = self.robot_name_init
            if from_group not in self.robot_voice:
                self.robot_voice[from_group] = self.robot_voice_init
            if from_group not in self.robot_language:
                self.robot_language[from_group] = self.robot_language_init
            if from_group not in self.robot_speed:
                self.robot_speed[from_group] = self.robot_speed_init
            if temp_memory["sender"]["user_id"]:
                from_person = str(temp_memory["sender"]["user_id"])
            else:
                from_person = "0000000000"
            for txt in temp_memory["message"]:
                if txt["type"] == "at":
                    if txt["data"]["qq"] != " ":
                        if txt["data"]["qq"] == os.getenv("QQ_NUM"):
                            all_txt.append({"from_person": from_person, "message": {"role": "user",
                                                                                    "content": "人物" + str(
                                                                                        from_person) + " @了人物" +
                                                                                               self.robot_name[
                                                                                                   from_group]}})
                        else:
                            all_txt.append({"from_person": from_person, "message": {"role": "user",
                                                                                    "content": "人物" + str(
                                                                                        from_person) + " @了人物 " + str(
                                                                                        txt["data"]["qq"])}})

                if txt["type"] == "reply":
                    all_txt.append({"from_person": from_person,
                                    "message": {"role": "user", "content": "人物" + str(from_person) + " 回复了消息"}})
                if txt["type"] == "text":
                    if txt["data"]["text"] != " ":
                        all_txt.append({"from_person": from_person, "message": {"role": "user", "content": "人物" + str(
                            from_person) + "说：" + str(txt["data"]["text"])}})

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
                        all_txt.append({"from_person": from_person, "message": {"role": "user", "content": "人物" + str(
                            from_person) + "说 ：" + str(vo_info)}})

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
                send_flag = True
                terminal = info["message"]["content"].replace("命令", "")
            if self.robot_name[from_group] in info["message"]["content"]:
                send_flag = True
                from_person = info["from_person"]
                self.memory.group_memory[from_group]["assistant_memory"].append(info["message"])
            if "联网查询" in info["message"]["content"]:
                web_search_flag = True
                from_person = info["from_person"]
                info["message"]["content"] = info["message"]["content"].replace(self.robot_name[from_group], "")
            self.memory.group_memory[from_group]["group_memory"].append(info["message"])
        if send_flag:
            self.message_answer(from_group, web_search_flag, master_flag, terminal, from_person)

    def message_answer(self, from_group, web_search_flag, master_flag, terminal, from_person):
        if master_flag:
            print("收到命令")
            if "出示答题过程" in terminal:
                send_message_dati(from_group)
            if "出示声音列表" in terminal:
                voice_list = self.t_k.voice_list.keys()
                voice_list_str = "\n"
                index = 0
                for voice in voice_list:
                    voice = str(voice)
                    index = index + 1
                    voice_list_str = voice_list_str + voice + " \n"
                    if index == 10:
                        send_info = {'type': 'text', 'data': {'text': "声音列表：" + voice_list_str}}
                        send_message_to_group(from_group, send_info)
                        index = 0
                        voice_list_str = "\n"
                send_info = {'type': 'text', 'data': {'text': "声音列表：" + voice_list_str}}
                send_message_to_group(from_group, send_info)

            if "出示角色列表" in terminal:
                name_list = role_dict.keys()
                name_list_str = "\n"
                for name in name_list:
                    name_list_str = name_list_str + name + " \n"

                send_info = {'type': 'text', 'data': {'text': "角色列表：" + name_list_str}}
                send_message_to_group(from_group, send_info)

            if "出示真名" in terminal:
                send_info = {'type': 'text', 'data': {'text': "我是" + self.robot_name[from_group]}}
                send_message_to_group(from_group, send_info)
            if "出示现在声音" in terminal:
                send_info = {'type': 'text', 'data': {'text': "现在声音是：" + str(self.robot_voice[from_group])+"\n现在的速度是："+str(self.robot_speed[from_group])+ "\n现在的语言是："+str(self.robot_language[from_group])}}
                send_message_to_group(from_group, send_info)

            if "重载语言" in terminal:
                language_name = terminal.replace("重载语言", "")
                for language in self.robot_language_list:
                    if language in language_name:
                        language_name = language
                        self.robot_language[from_group] = language_name
                        send_info = {'type': 'text', 'data': {'text': "语言已经切换为：" + language_name}}
                        send_message_to_group(from_group, send_info)
                        return 0
                send_info = {'type': 'text', 'data': {'text': "语言不存在：" + language_name}}
                send_message_to_group(from_group, send_info)


            if "重载声音" in terminal:
                voice_name = terminal.replace("重载声音", "")
                voice_name = voice_name.replace(" ", "").replace("\n", "").replace("\r", "").replace("\t", "").replace("人物", "").replace("说", "").replace("1", "").replace("2", "").replace("3", "").replace("4", "").replace("5", "").replace("6", "").replace("7", "").replace("8", "").replace("9", "").replace("0", "").replace(":","").replace("：","").replace("。","").replace(".","").replace("，","").replace(",","").replace("？","").replace("?","").replace("！","").replace("!","").replace("“","").replace("”","").replace("\"","").replace("；","").replace(";","").replace("（","").replace("）","").replace("(","").replace(")","").replace("【","").replace("】","").replace("[","").replace("]","").replace("《","").replace("》","").replace("<","").replace(">","").replace("、","").replace("/","").replace("\\","").replace("|","").replace(" ","")
                voice_list = self.t_k.voice_list.keys()
                print("简化后的声音名字：")
                print(voice_name)

                for voice in voice_list:
                    print(voice)
                    print("\n")
                    if voice_name in voice:
                        self.robot_voice[from_group] = voice
                        send_info = {'type': 'text', 'data': {'text': "重载成功"}}
                        send_message_to_group(from_group, send_info)
                        return None
                send_info = {'type': 'text', 'data': {'text': "重载失败，没有找到该声音"}}
                send_message_to_group(from_group, send_info)

            if "重载角色" in terminal:
                name_list = role_dict.keys()
                for name in name_list:
                    if name in terminal:
                        self.robot_name[from_group] = name
                        send_info = {'type': 'text', 'data': {'text': "重载成功"}}
                        send_message_to_group(from_group, send_info)
                        self.memory.reload_robot_memory(from_group, self.robot_name[from_group])
                        return None
                send_info = {'type': 'text', 'data': {'text': "重载失败，没有找到该角色"}}
                send_message_to_group(from_group, send_info)

        else:
            if os.getenv("CONNECT_TO_INTERNET") == "True" and web_search_flag and self.memory.authority[from_person][
                "use_ai_web_search"]:
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
                        self.memory.group_memory[from_group_lan]["group_memory"].append(
                            {"role": "assistant", "content": str(ikaros_answer)})
                        if self.memory.authority[from_person]["use_ai_voice"]:
                            t1 = threading.Thread(target=send_file_in_japanese_to_group,
                                                  args=(from_group_lan, ikaros_answer, self.t_k, self.robot_language[from_group_lan], self.robot_speed[from_group_lan], self.robot_voice[from_group_lan]))
                            t1.start()

                    else:
                        send_message_to_group(from_group_lan,
                                              {'type': 'text', 'data': {'text': "机器人连接openai出现了一些问题，请联系管理员"}})

                if self.memory.authority[from_person]["use_ai"]:
                    t2 = threading.Thread(target=my_thread,
                                          args=(self.memory.group_memory[from_group], from_group))
                    t2.start()
                else:
                    send_message_to_group(from_group,
                                          {'type': 'text', 'data': {'text': str(from_person) + "您的权限不足无法使用AI"}})
