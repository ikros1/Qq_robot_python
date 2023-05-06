import os

import requests
import urllib.parse
from dotenv import load_dotenv

load_dotenv()


def encode_text(text):
    encoded_text = urllib.parse.quote(text)
    return encoded_text


def check_txt(txt_data):
    url = "https://aip.baidubce.com/rest/2.0/solution/v1/text_censor/v2/user_defined?access_token=" + get_access_token()

    payload = 'text=' + encode_text(txt_data)
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    return_data = response.json()
    # 转为字典
    return_data = dict(return_data)
    print(return_data)
    return parse_json(return_data)


def parse_json(json_data):
    reasons = []
    # 判断是否存在key
    if "conclusion" in json_data:
        if json_data["conclusion"] == "不合规":
            for item in json_data["data"]:
                if item["conclusion"] == "不合规":
                    reasons.append(item["msg"])
                    for hit in item["hits"]:
                        words = hit.get("words", [])
                        if len(words) > 0:
                            reason = "违规词: {}".format(",".join(words))
                            reasons.append(reason)

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
    success, data = (check_txt("我永远爱你"))
    if success:
        print("合规")
    else:
        print("不合规")
        print(data)
