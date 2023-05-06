import base64
import os

import requests
import urllib.parse
from dotenv import load_dotenv

load_dotenv()


def encode_text(text):
    encoded_text = urllib.parse.quote(text)
    return encoded_text


def check_img(img_url):
    url = "https://aip.baidubce.com/rest/2.0/solution/v1/img_censor/v2/user_defined?access_token=" + get_access_token()
    # 对图片进行base64编码
    payload = 'imgUrl=' + img_url
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    return_data = response.json()
    # 转为字典
    return_data = dict(return_data)
    # print(return_data)
    return parse_json(return_data)


def parse_json(json_data):
    reasons = []
    if json_data["conclusion"] == "不合规":
        for item in json_data["data"]:
            if item["conclusion"] == "不合规":
                reasons.append(item["msg"])
    if len(reasons) > 0:
        return False, reasons
    else:
        return True, "合规"


def get_access_token():
    """
    使用 AK，SK 生成鉴权签名（Access Token）
    :return: access_token，或是None(如果错误)
    """
    API_KEY = os.getenv("API_KEY")
    SECRET_KEY = os.getenv("SECRET_KEY")
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": API_KEY, "client_secret": SECRET_KEY}
    return str(requests.post(url, params=params).json().get("access_token"))


if __name__ == '__main__':

    success, data = (check_img(
        "https://c2cpicdw.qpic.cn/offpic_new/1692298249//1692298249-1177498474-48D321ADA75FFC43BDB67C12C6C53B04/0?term=2&is_origin=0"))
    if success:
        print("合规")
    else:
        print("不合规")
        print(data)
