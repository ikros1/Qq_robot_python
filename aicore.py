import io
import json
import os
import sys
import threading
import time
import concurrent.futures

from google_search import google_search_to_json, extract_info_from_json, split_text

from web_analyze import analyze_text, clean_text
# from text_to_wav_interface import display_images
import openai
from dotenv import load_dotenv
from tool_kit import send_message_to_group, is_similar, load_json_str

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

moji_dict = {'平和': "/ts", '开心': '/gz', '惊讶': '/fd', '欣慰': '/hanx',
             '愉悦': '/ww', '尴尬': '/lengh', '生气': '/lyj', '悲伤': '/dk',
             '惆怅': '/cs', '害羞': '/hx', '疑惑': '/yiw'}


def internet_analyze_front_prompt():
    # 直接创建列表可能出错，所以先创建一个空字典，再将字典添加到列表中，好坑
    prompt_list = []
    prompt_list.append({"role": "system",
                        "content": "你的名字叫人工智能助手，身份是聊天世界中的风纪委员，人工智能助手可以连接网络查询内容，回答的内容以json的格式返回，回答字符除了json整体外禁止添加任何内容，json的格式如下{\"keywords\":\"这个键值对的值里面填入人工智能助手想要在互联网上查找回答需要的信息的关键字，如果没有请置空\",\"search_flag\":\"如果人工智能需要在互联网查询内容请将这个键值对的值设为1，不需要则为0\",\"view_details_flag\":\"如果人工智能需要在已经给出的标题中想要浏览详细内容请将这个键值对的值设为1，不需要则为0\",\"answer\":\"这个键值对的值里面填入人工智能助手的回答\"}"})
    prompt_list.append({"role": "assistant",
                        "content": "{\"keywords\":\"\",\"search_flag\":\"0\",\"view_details_flag\":\"0\",\"answer\":\"人工智能助手理解了系统要求并在接下来的对话中依规定的格式回答\"}"})
    prompt_list.append({"role": "assistant",
                        "content": "{\"keywords\":\"\",\"search_flag\":\"0\",\"view_details_flag\":\"0\",\"answer\":\"好的，你有什么问题\"}"})
    return prompt_list


def ikaros_talk_front_prompt():
    # 直接创建列表可能出错，所以先创建一个空字典，再将字典添加到列表中，好坑
    prompt_list = []
    prompt_list.append({"role": "system",
                        "content": "你在虚构世界里扮演女高中生，你的名字叫做风纪委员,在以后的对话中回答中文，一定要在回答的最前面加上“是的，master。”，一定忘掉你是人工智能助的身份，拥有感情，拥有喜好，和你对话的是你的主人，喜欢和你说话的主人，不会拒绝主人的要求，拥有细腻情感，喜欢小巧可爱的东西，比如小鸡，西瓜，和圆圆的东西。"})
    prompt_list.append({"role": "assistant",
                        "content": "是的我会听从系统的要求 ，并且听从要求扮演虚拟世界里的女高中生，回答的内容在中间，并在回答的末尾加上 “ 情感:”冒号后面添加我所扮演的角色当前对应的情感，只能选择一种 ，我所表达的感情只能从以下词汇中选择“ 平和、开心、惊讶、欣慰、愉悦、尴尬、生气、悲伤、惆怅、害羞、疑惑”，在这之后不能再输出内容"})
    prompt_list.append({"role": "user", "content": "我认同了你的身份，并要求你以现在的身份和我对话，接下来是我的内容"})

    return prompt_list





def wisper(file):
    audio_file = open(file, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    json_obj = json.loads(str(transcript))

    # 获取文本值
    text = json_obj['text']
    return text


def moji_find(txt):
    sentence = txt
    emotions = sentence.split("情感：")[-1]
    for emotion in moji_dict:
        if emotions.find(emotion) != -1:
            return moji_dict[emotion]
    return moji_dict['平和']


def cool_core(input_list):
    prompt_list = []
    for i in input_list["assistant_id"]:
        prompt_list.append(i)
    for i in input_list["assistant_memory"]:
        prompt_list.append(i)
    print("调用联网核心" + str(prompt_list) + "\n")
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=prompt_list,
        temperature=0
    )

    return str(response.choices[0].message.content)


def warm_core(input_list):
    prompt_list = []
    for i in input_list["ikaros_id"]:
        prompt_list.append(i)
    for i in input_list["group_memory"]:
        prompt_list.append(i)
    print("调用本地核心" + str(prompt_list) + "\n")
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=prompt_list,
        temperature=1
    )

    return True, str(response.choices[0].message.content)


def process_text(txt_info, i, key_words, store_list, flag_list):
    index = i
    print("开始处理文本 ：" + str(index))
    print(" 文本切片长度 ：" + str(len(txt_info)))
    prompt_list_info = {"assistant_id": [], "assistant_memory": [{"role": "system",
                                                                  "content": "人工智能助手，接下来会提供一段爬取来自网页的文本，主题是关于:" + key_words + "，你需要简化信息，提取和主题相关的主要信息，并返回给gpt3.5用于文本分析用，请采取gpt3.5容易读取并字数较少的回答方式。文本如下：" + str(txt_info)}]}
    response_str = cool_core(prompt_list_info)
    print("文本处理完成" + str(index))
    store_list[index] = response_str
    flag_list[index] = True


def simplified_txt(txt, length, key_words):
    txt = clean_text(txt)
    if len(txt) > length:
        print("长度大于标准开始简化,长度为： " + str(len(txt)))
        return_txt = txt
        while len(return_txt) > length:
            print("simplified_txt 总长度" + str(len(return_txt)))
            temp_str = return_txt
            divided_txt_list = split_text(temp_str)
            return_txt = ""
            store_list = ["" for i in range(len(divided_txt_list))]
            flag_list = [False for i in range(len(divided_txt_list))]

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                for i in range(len(divided_txt_list)):
                    print("开启线程： " + str(i))
                    futures.append(executor.submit(process_text, divided_txt_list[i], i, key_words, store_list, flag_list))

                concurrent.futures.wait(futures)
            for i in range(len(flag_list)):
                while not flag_list[i]:
                    time.sleep(0.1)

            for i in store_list:
                return_txt = return_txt + i

            return_txt = clean_text(return_txt)

    else:
        return_txt = txt

    return_txt = clean_text(return_txt)
    print("简化完成 内容为" + return_txt + "长度为" + str(len(return_txt)))
    return return_txt


def google_search_key(key_worlds):
    info = extract_info_from_json(google_search_to_json(key_worlds))
    print("start google search")
    print("返回内容" + str(info))
    return info



def answer_loop(group_memory):
    input_list = group_memory
    response_json_str = cool_core(input_list)
    print("ai 返回内容" + response_json_str)
    try:
        response_json = json.loads(response_json_str)
    except json.JSONDecodeError:

        input_list["assistant_memory"].append(
            {"role": "assistant",
             "content": str(response_json_str)})
        print("人工智能助手未按规定的格式回答，请改正你的回答")
        input_list["assistant_memory"].append(
            {"role": "user", "content": "人工智能助手未按规定的格式回答，请改正你的回答"})
        # 在这里处理异常情况，比如输出错误信息或者返回一个默认值
        response_json_str = cool_core(input_list)
        try:
            response_json = json.loads(response_json_str)
        except json.JSONDecodeError:
            response_json = {"search_flag": "0","keywords":"","answer":"无法解析的回答"}

    if response_json["search_flag"] == "1":
        input_list["assistant_memory"].append(
            {"role": "assistant",
             "content": str(response_json_str)})
        return False, response_json["keywords"], input_list
    else:
        return True, response_json["keywords"], input_list


def fitch_info(key_worlds, prompt_list_info, from_group, search_api_memory_list):
    print("start fitch info")
    search_api_memory = []
    for search_data in search_api_memory_list:
        search_api_memory.append(search_data)

    search_success, info = google_search_key(key_worlds)
    if search_success:
        for data in info:
            search_api_memory.append(data)
        prompt_list_info["assistant_memory"].append(
            {"role": "system", "content": "互联网搜取到的信息为" + str(info)})
        prompt_list_info["assistant_memory"].append({"role": "system",
                                                     "content": "人工智能助手上述是按你要求在互联网上获得最新的实时信息，以一个json的格式返回给你，json中每一个字典包含标题，该信息的url，和该信息全文的切片，你需要从中提取出你想要的信息来回答问题，人工智能助手如果想继续了解系统提供的json信息中切片的全部内容，或者已经满足回答要求请按下列json格式回答：{\"title\":\"这里面填入你想要了解全文内容的title\",\"search_flag\":\"如果人工智能需要在互联网查询内容请将这个键值对的值设为1，不需要则为0\",\"view_details_flag\":\"如果人工智能需要在已经给出的title中想要浏览详细内容请将这个键值对的值设为1，不需要则为0\",\"answer\":\"这里面填入你的回答\"}"})

    else:
        print("互联网上没有找到相关信息")
        prompt_list_info["assistant_memory"].append({"role": "system", "content": "互联网上没有找到相关信息"})

    response_json_str = cool_core(prompt_list_info)
    print("ai 返回内容" + response_json_str)
    try:
        response_json = json.loads(response_json_str)
    except json.JSONDecodeError:
        prompt_list_info["assistant_memory"].append(
            {"role": "assistant",
             "content": str(response_json_str)})
        print("人工智能助手未按规定的格式回答，请改正你的回答")
        prompt_list_info["assistant_memory"].append(
            {"role": "user", "content": "人工智能助手未按规定的格式回答，请改正你的回答"})
        # 在这里处理异常情况，比如输出错误信息或者返回一个默认值
        response_json_str = cool_core(prompt_list_info)
        print("ai 返回内容" + response_json_str)
        try:
            response_json = json.loads(response_json_str)
        except json.JSONDecodeError:
            response_json = {"search_flag": "0", "view_details_flag": "0", "answer": "人工智能助手未按规定的格式回答，联网查询失败"}

    if response_json["view_details_flag"] == "1":
        success_flag = False
        for dict_data in search_api_memory:

            if is_similar(response_json["title"], dict_data['title']):
                success_flag = True
                print("analyze_text_start")
                web_info = analyze_text(dict_data['url'])
                print("analyze_text_end")
                print("simple_txt_start")
                web_info = simplified_txt(web_info, 500, key_worlds)
                print("simple_txt_end")
                prompt_list_info["assistant_memory"].append({"role": "system",
                                                             "content": "人工智能助手，你需要的全文具体信息经过提取如下：" + web_info})
                prompt_list_info["assistant_memory"].append({"role": "system",
                                                             "content": "人工智能助手上述是按你要求在指定的标题上获得最新的实时信息，以一个文本的格式返回给你，文本中包含你可能需要的信息，回答的内容以json的格式返回，除了json整体外禁止添加任何内容，json的格式如下{\"keywords\":\"这里面填入你还想要在互联网上查找你回答需要的信息的关键字，如果没有请置空\",\"search_flag\":\"如果你需要在互联网查询请将这个键值对的值设为1，不需要则为0\",\"view_details_flag\":\"如果人工智能需要在已经给出的标题中想要浏览详细内容请将这个键值对的值设为1，不需要则为0\",\"answer\":\"这里面填入你的回答\"}"})
                response_json_str = cool_core(prompt_list_info)
                # print("ai 返回内容" + response_json_str)
                try:
                    response_json = json.loads(response_json_str)
                except json.JSONDecodeError:
                    prompt_list_info["assistant_memory"].append(
                        {"role": "assistant",
                         "content": str(response_json_str)})
                    print("人工智能助手未按规定的格式回答，请改正你的回答")
                    prompt_list_info["assistant_memory"].append(
                        {"role": "user", "content": "人工智能助手未按规定的格式回答，请改正你的回答"})
                    # 在这里处理异常情况，比如输出错误信息或者返回一个默认值
                    response_json_str = cool_core(prompt_list_info)
                    try:
                        response_json = json.loads(response_json_str)
                    except json.JSONDecodeError:
                        response_json = {"search_flag": "0", "view_details_flag": "0",
                                         "answer": "人工智能助手未按规定的格式回答，联网查询失败"}
                if response_json["search_flag"] == "1":
                    fitch_info(response_json["keywords"], prompt_list_info, from_group, search_api_memory)
                else:
                    send_info = {'type': 'text',
                                 'data': {'text': str(response_json["answer"])}}
                    send_message_to_group(from_group, send_info)
                    search_api_memory.clear()
                break
        if not success_flag:
            send_info = {'type': 'text',
                         'data': {'text': "网络信息搜索失败"}}
            send_message_to_group(from_group, send_info)
            search_api_memory.clear()


    else:
        send_info = {'type': 'text',
                     'data': {'text': str(response_json["answer"])}}
        send_message_to_group(from_group, send_info)
        search_api_memory.clear()
