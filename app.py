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
    video, video_path, video_file_name = save_file.run(video_file)
    return render_template('loading.html', video=video, video_path=video_path, video_file_name=video_file_name)

@app.route('/complete', methods=['POST'])
def analysis():
    if request.method == "POST":
        video = request.form.get('video')
        video_path = request.form.get('video_path')
        video_file_name = request.form.get('video_file_name')

        # 동영상을 1FPS로 캡쳐하여 이미지 데이터로 변환
        video_to_imgs.run(video, video_file_name)

        # 캡쳐한 이미지 데이터에서 Avatar Detection 및 Cropping
        crop_avatars.run()
        # checkpoint = 'object_detection/BEST_checkpoint_ssd300.pth.tar'
        # checkpoint = torch.load(checkpoint, map_location=device)
        # start_epoch = checkpoint['epoch'] + 1
        # best_loss = checkpoint['loss']
        # print('\nLoaded checkpoint from epoch %d. Best loss so far is %.3f.\n' % (start_epoch, best_loss))

        # model = checkpoint['model']
        # model = model.to(device)
        # model.eval()
        # num=0

        # for i in range(6):
        #     img_path = './static/imgs/RobloxVideo/' + str(i+1) + '.jpg'
        #     original_image = Image.open(img_path, mode='r')
        #     original_image = original_image.convert('RGB')
        #     try:
        #         os.mkdir("./static/detection/")
        #     except OSError:
        #         None
        #     detect_path="./static/detection/" + str(num) + ".jpg"
        #     num+=1
        #     detect.detect(original_image, min_score=0.2, max_overlap=0.1, top_k=1000, model=model).save(detect_path, "JPEG")

        # Cropped 이미지 데이터에서 특징점 추출 및 Clustering

        # 유저 아바타와 접촉한 타 아바타 개수와 그에 비례하는 리워드 금액 반환

        return render_template('complete.html', video=video_path)

# 동영상 분석 결과(Clustering 결과 그래프) 반환 함수
@app.route('/result', methods=['POST'])
def result():
    if request.method == "POST":
        video = request.form.get('video')
        return render_template('result.html', video=video)

if __name__=="__main__":
  app.run(debug=True)