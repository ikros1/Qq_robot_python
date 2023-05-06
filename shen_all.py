# 建立类 shen
# 用于处理所有审核数据
import subprocess
import time
import urllib
from shen_txt import check_txt
from shen_img import check_img
from tool_kit import send_message_to_person, send_message_to_group, send_message_dati
from wisper_to_text import voice_to_text
from pydub import AudioSegment
import os


def send_message(from_group, send_to_person, reason, send_flag, data):
    if send_flag:
        # print("发送审核违规信息")
        send_to_manager_qq = os.getenv("MANAGER_QQ")
        # print(self.reason)
        reason_txt = str(reason)
        # 去除所有中括号
        reason_txt = reason_txt.replace("[", "").replace("]", "")
        moji_str = "内个内个，您的内容" + reason_txt + "，请，请注意一下（要求太严格的话，不要揍我 嘤嘤嘤）"
        reason_info = [{'type': 'at', 'data': {'qq': send_to_person}},
                       {'type': 'text', 'data': {'text': moji_str}}]
        reason_info2 = [{'type': 'at', 'data': {'qq': send_to_person}},
                        {'type': 'text', 'data': {
                            'text': "违规信息来自群聊" + str(from_group) + "来自qq：" + str(
                                send_to_person) + " 违规原因：" + str(reason)}}]

        send_message_to_person(send_to_manager_qq, reason_info2)
        send_message_to_person(send_to_manager_qq, data["message"])
        send_message_to_group(from_group, reason_info)


def receive_data(data):
    from_group = None
    send_to_person = None
    reason = []
    send_flag = False
    if data["post_type"] == "message" and data["message_type"] == "group":
        from_group = data["group_id"]
        send_to_person = data["user_id"]
        for txt in data["message"]:
            if txt["type"] == "text":
                if txt["data"]["text"] != " ":
                    if "入群答题方式" in txt["data"]["text"]:
                        send_message_dati(from_group)
            if txt["type"] == "text":
                if txt["data"]["text"] != " ":
                    success, info = check_txt(txt["data"]["text"])
                    if not success:
                        print("审核文字失败")
                        print(info)
                        send_flag = True
                        reason.append(info)

            if txt["type"] == "image":
                success, info = check_img(txt["data"]["url"])
                if not success:
                    # print("审核图片失败")
                    print(info)
                    send_flag = True
                    reason.append(info)

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
                print(vo_info)

                if vo_info != " ":
                    success, info = check_txt(vo_info)
                    if not success:
                        # print("审核文字失败")
                        print(info)
                        send_flag = True
                        reason.append(info)

        send_message(from_group=from_group, send_to_person=send_to_person, reason=reason, send_flag=send_flag,
                     data=data)



