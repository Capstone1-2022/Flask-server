import cv2
import os

def run(video):
    # 기존 파일 삭제
    for file in os.scandir("./static/imgs"):
        os.remove(file.path)

    for file in os.scandir("./static/detection"):
        if(os.path.isfile(file.path)):
            os.remove(file.path)

    for file in os.scandir("./static/detection/crop"):
        os.remove(file.path)

    # 새 파일 저장 시작
    video = cv2.VideoCapture(video)
    fps = video.get(cv2.CAP_PROP_FPS)
    cnt = 1

    if not os.path.exists("./static/imgs"):
        os.mkdir("./static/imgs/")

    while(video.isOpened()):
        ret, img = video.read()
        if(int(video.get(1)) % int(fps) == 0):
            num = str(cnt)
            imgfile_name = "./static/imgs/" + num + ".jpg"
            cv2.imwrite(imgfile_name, img)
            cnt += 1
        if(ret == False): break

    video.release()

    return cnt-1