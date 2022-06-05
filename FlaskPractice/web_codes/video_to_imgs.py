import cv2
import os

def run(video, video_file_name):

    video = cv2.VideoCapture(video)
    fps = video.get(cv2.CAP_PROP_FPS)
    cnt = 1

    if not os.path.exists("./static/imgs/" + video_file_name):
        os.mkdir("./static/imgs/" + video_file_name)

    while(video.isOpened()):
        ret, img = video.read()
        if(int(video.get(1)) % int(fps) == 0):
            num = str(cnt)
            imgfile_name = "./static/imgs/" + video_file_name + "/" + num + ".jpg"
            cv2.imwrite(imgfile_name, img)
            cnt += 1
        if(ret == False): break

    video.release()