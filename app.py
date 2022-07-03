import text_detect
import crop_avatars
import myresnet.load_resnet
import word_cloud

from web_codes import save_file
from web_codes import video_to_imgs
from myresnet.load_resnet import *
from text_detect import *
from flask import Flask, render_template, request
from word_cloud import *

app = Flask(__name__)

@app.route('/',methods=['GET'])
def home():
    if request.method == "GET":
        return render_template('home.html')

@app.route('/loading', methods=['POST'])
def loading():
    video_file = request.files['file']
    video, video_path = save_file.run(video_file)
    return render_template('loading.html', video=video, video_path=video_path)

@app.route('/complete', methods=['POST'])
def analysis():
    if request.method == "POST":
        video = request.form.get('video')
        video_path = request.form.get('video_path')

        # 동영상을 1FPS로 캡쳐하여 이미지 데이터로 변환
        img_num = video_to_imgs.run(video)

        # 캡쳐한 이미지 데이터에서 Avatar Detection 및 Cropping
        crop_avatars.run(img_num)

        # Cropped 이미지 데이터에서 특징점 추출 및 Clustering
        origin = myresnet.load_resnet.run()
        count = origin - 1

        # Cropped 이미지 데이터에서 닉네임 Detection 및 Clustering
        # text_count = text_detect.run()
        word_cloud.run()
        

        # 유저 아바타와 접촉한 타 아바타 개수와 그에 비례하는 리워드 금액 반환(1명당 50원)
        estimate = round((text_count + count)/2)
        reward = estimate * 50

        return render_template('complete.html', video=video_path, count=count, origin=origin, reward=reward, text_count=text_count, estimate=estimate)

# 동영상 분석 결과(Clustering 결과 그래프) 반환 함수
@app.route('/result', methods=['POST'])
def result():
    if request.method == "POST":
        video = request.form.get('video')
        count = request.form.get('count')
        origin = request.form.get('origin')
        text_count = request.form.get('text_count')
        estimate = request.form.get('estimate')
        reward = request.form.get('reward')
        return render_template('result.html', video=video, count=count, origin=origin, text_count=text_count, estimate=estimate, reward=reward)

if __name__=="__main__":
  app.run(host='0.0.0.0')