import os
# import model_dy
import time

import socketio
from flask import Flask, render_template

import time
from threading import Thread
from flask import Flask, request, render_template, redirect, url_for, g, send_from_directory

from file_utis import clear_folder,is_film_empty
from model_dy import make_image
from s3 import reTrain

app = Flask(__name__)

@app.before_request
def bef():
    setattr(g, "isShow", False)
    setattr(g, "isRetrainSucc", False)

from flask import request, redirect, url_for

from flask import request, redirect, url_for

# 登录页面
@app.route("/", methods=["GET", "POST"])
def aa1():
    if request.method == "POST":
            return redirect(url_for('aa0'))  # 登录成功，重定向到主页
        # else:
        #     error = "Invalid username or password."  # 设置错误信息
        #     return render_template('index2.html', error=error)  # 重新渲染登录页面，显示错误信息
    else:
        return render_template('index2.html')  # GET请求时显示登录页面

# 主页界面
@app.route("/test")
def aa0():
    if request.method == "POST":
        # 处理 POST 请求的逻辑
        pass
    else:
        # 处理 GET 请求的逻辑
        return render_template('test.html')

# 注册
@app.route("/register")
def aa2():
    return render_template('register.html')


@app.route("/q")
def aa():
    return render_template('index.html',isShow = g.isShow,isRetrainSucc=g.isRetrainSucc )

# 自定义
@app.route('/diy')
def diy():
    return render_template('dd.html')

# 风格1
@app.route('/style1')
def style1():
    return render_template('style1.html')


@app.route('/trans', methods=['get','POST'])
def process_image():
    if request.method=='GET':

        return render_template('index.html',isShow = g.isShow,isRetrainSucc=g.isRetrainSucc )
    if request.method=='POST':
        g.isShow = True
        files = request.files.getlist('ImageFiles')
        upload_folder = f'static/original/1/a'
        for file in files:
            file_path = os.path.join(upload_folder, file.filename)
            file.save(file_path)

        file2 = request.files['StyleImageFile']
        upload_folder = 'static/style'
        file_path2 = os.path.join(upload_folder, "xingkong.jpg")
        file2.save(file_path2)

        make_image()
        image_folder = 'static/inputs'
        while(1):
            if is_film_empty(image_folder) ==False:
                break
            time.sleep(2)

        isShow = True
        return render_template('index.html',isShow = g.isShow,isRetrainSucc=g.isRetrainSucc )

@app.route('/trans_style1', methods=['get','POST'])
def process_image_style1():
    if request.method=='GET':

        return render_template('style1.html',isShow = g.isShow,isRetrainSucc=g.isRetrainSucc )
    if request.method=='POST':
        g.isShow = True
        upload_folder = f'flasktest/static/original/1/a/'
        os.makedirs(upload_folder, exist_ok=True) 
        files = request.files.getlist('ImageFiles')
        
        for file in files:
            file_path = os.path.join(upload_folder, file.filename)
            file.save(file_path)

        # file2 = request.files['StyleImageFile']
        # upload_folder = 'static/style/style1'
        # file_path2 = os.path.join(upload_folder, "xingkong.jpg")
        # file2.save(file_path2)
        style_image=r'flasktest\static\style\style1\xingkong.jpg'
        make_image(style_image)
        image_folder = 'flasktest\static/inputs'
        while(1):
            if is_film_empty(image_folder) ==False:
                break
            time.sleep(2)

        isShow = True
        return render_template('style1.html',isShow = g.isShow,isRetrainSucc=g.isRetrainSucc )


@app.route('/show_image')
def show():
    g.isShow=False
    image_folder = 'flasktest/static/inputs'
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    return render_template('index.html', image_files=image_files,isShow = g.isShow,isRetrainSucc=g.isRetrainSucc )

@app.route('/download')
def download():
    # 获取static目录下所有图片文件的列表
    UPLOAD_FOLDER = app.config['UPLOAD_FOLDER']
    image_files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.endswith(('.jpg', '.png', '.gif'))]

    if image_files:
        for image in image_files:
            return send_from_directory(app.config['UPLOAD_FOLDER'], image)
    else:
        return "no picture"

    # 这里假设你的上传目录是 'static/'
app.config['UPLOAD_FOLDER'] = 'static/inputs'

@app.route('/reSet',methods=['POST'])
def reset():
    L= {
        'base': request.form.get('base'),
        'style_weight': request.form.get('style_weight'),
        'content_weight': request.form.get('content_weight'),
        'tv_weight': request.form.get('tv_weight'),
        'epochs': request.form.get('epochs'),
        'batch_size': request.form.get('batch_size'),
        'width': request.form.get('width'),
        'verbose_hist_batch': request.form.get('verbose_hist_batch'),
        'verbose_image_batch': request.form.get('verbose_image_batch')
    }
    time.sleep(100)
    # reTrain(L)
    g.isRetrainSucc = True
    return render_template('index.html',isShow = g.isShow,isRetrainSucc=g.isRetrainSucc )


@app.route('/clear')
def clear_():
    dir1 = 'static/inputs'
    dir2 = 'static/original/1/a'
    dir3 = 'static/style'
    g.isRetrainSucc = False
    g.isShow = False
    clear_folder(dir1)
    clear_folder(dir2)
    clear_folder(dir3)
    return redirect(url_for('aa'))

if __name__ == '__main__':
    app.run (debug=True)


