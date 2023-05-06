import os

import requests
import json
import threading
import subprocess
from dotenv import load_dotenv
import json

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
CUSTOM_SEARCH_ENGINE_ID = os.getenv("CUSTOM_SEARCH_ENGINE_ID")


def google_search_to_json(input_keyword):
    # 去除input_keyword除了字符和数字以外的所有字符
    input_keyword = ''.join([i for i in input_keyword if i.isalnum()])
    print("input_keyword: " + input_keyword)
    url = "https://www.googleapis.com/customsearch/v1?key=" + GOOGLE_API_KEY + "&q=\"" + input_keyword + "\"&cx=" + CUSTOM_SEARCH_ENGINE_ID + "&start=1&num=5"
    # 发送请求
    response = requests.request("GET", url)
    # 转为字典
    return_data = json.loads(response.text)
    print("return_data: "+str(return_data))
    return return_data


def extract_info_from_json(json_data):
    if 'items' not in json_data:
        return False, []
    else:
        items = json_data['items']
        info_list = []
        for item in items:
            info = {'title': item['title'], 'url': item['link'], 'snippet': item['snippet']}
            info_list.append(info)
        return True, info_list




def split_text(text, chunk_size=1500):
    """
    Split a text into chunks of size chunk_size, trying to split near a sentence-ending punctuation.
    :param text: A string representing the text to split.
    :param chunk_size: An integer representing the desired size of each chunk.
    :return: A list of strings representing the split chunks.
    """
    chunks = []
    while len(text) > chunk_size:
        split_index = text.rfind(".", 1450, chunk_size)  # Look for the last sentence-ending punctuation before chunk_size.
        if split_index == -1:
            split_index = text.rfind("。", 1450, chunk_size)
        if split_index == -1:
            split_index = text.rfind("？", 1450, chunk_size)
        if split_index == -1:
            split_index = text.rfind("?", 1450, chunk_size)
        if split_index == -1:
            split_index = text.rfind("！", 1450, chunk_size)
        if split_index == -1:
            split_index = text.rfind("!", 1450, chunk_size)
        if split_index == -1:
            split_index = text.rfind("，", 1450, chunk_size)
        if split_index == -1:
            split_index = text.rfind(",", 1450, chunk_size)
        if split_index == -1:  # If no punctuation is found, just split at chunk_size.
            split_index = chunk_size
        chunks.append(text[:split_index].strip())
        text = text[split_index:].strip()
    chunks.append(text)  # Append the remaining text as the last chunk.
    return chunks

