import os
from werkzeug.utils import secure_filename

def run(video_file):
    # 기존 파일 삭제
    for file in os.scandir("./static/videos"):
        os.remove(file.path)

    # 새 파일 저장 시작
    video_file.save(os.path.join("./static/videos", secure_filename(video_file.filename)))
    video = "./static/videos" + "/" + secure_filename(video_file.filename)
    video_path = video

    return video, video_path