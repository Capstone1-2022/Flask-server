from web_codes import save_file
from web_codes import video_to_imgs
import crop_avatars

from flask import Flask, render_template, request

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
        exec(open("resnet/load_resnet.py").read())

        # 유저 아바타와 접촉한 타 아바타 개수와 그에 비례하는 리워드 금액 반환

        return render_template('complete.html', video=video_path)

# 동영상 분석 결과(Clustering 결과 그래프) 반환 함수
@app.route('/result', methods=['POST'])
def result():
    if request.method == "POST":
        video = request.form.get('video')
        return render_template('result.html', video=video)

if __name__=="__main__":
  app.run(host='0.0.0.0')