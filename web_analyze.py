import requests
from bs4 import BeautifulSoup
import nltk
import re


def clean_text(text):
    # 去除UTF-8违法字符
    cleaned_text = text.encode('utf-8', 'ignore').decode('utf-8')
    # 去除连续多余2个的回车
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
    # 去除多余的空格
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    return cleaned_text


def analyze_text(url):
    print("start url_web analyze")
    # 下载nltk所需的数据
    nltk.download('punkt')
    nltk.download('stopwords')

    # 发送HTTP请求并获取网页内容
    response = requests.get(url)
    html_content = response.content

    # 使用BeautifulSoup库解析HTML内容
    soup = BeautifulSoup(html_content, 'html.parser')

    # 提取标题、正文和日期
    body = soup.get_text()

    # 使用nltk库进行文本处理
    stopwords = nltk.corpus.stopwords.words('english')
    words = nltk.word_tokenize(body.lower())
    words_cleaned = [word for word in words if word.isalnum() and word not in stopwords]
    body_cleaned = ' '.join(words_cleaned)

    # 整合标题、正文和日期到一篇文章中
    article = body

    article = clean_text(article)

    print("analyze info : " + article)

    return article
