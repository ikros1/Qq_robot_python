import logging
import zipfile
from datetime import datetime
from flask import Response, make_response, jsonify, send_from_directory, request
from flask import Flask
from flask import render_template
import time
import json
from robot import Robot


def json_to_dict(json_str):
    return json.loads(json_str)


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    filename='app.log',
                    filemode='w')

app = Flask(__name__)
app.config['PROPAGATE_EXCEPTIONS'] = True
app.config['SECRET_KEY'] = 'AASDFASDF'

robot = Robot()


# app支持局域网其他设备访问

def app_start(port, debug):
    app.run(host='0.0.0.0', port=port, debug=debug, use_reloader=False)


@app.route('/hello-robot', methods=["POST"])
def hello_robot():
    # 接受前端传来的数据
    data = request.get_json()
    print(data)
    #dict_data = json_to_dict(data)
    # 将数据传入机器人类中
    robot.receive_data(data)
    response = make_response('0', 200)
    return response


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["file"]
    if file:
        filename = datetime.now().strftime("%Y%m%d%H%M%S") + "." + file.filename.rsplit(".", 1)[1]
        file.save("usr_upload_dir/" + filename)
        return {"message": "文件上传成功,开始检测"}
    else:
        return {"message": "文件上传失败,请刷新页面重新开始"}


@app.route('/download')
def download(filenames=None, dump_names=None):
    # 创建一个zip文件
    zip_file = zipfile.ZipFile('downloaded_video_save_dir/files.zip', 'w')
    # 将每个文件添加到zip文件中
    for filename in filenames:
        zip_file.write(filename)
    for filename in dump_names:
        zip_file.write(filename)
    # 关闭zip文件
    zip_file.close()
    return send_from_directory('downloaded_video_save_dir/files.zip',
                               "files.zip")




