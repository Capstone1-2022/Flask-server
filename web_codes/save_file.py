import os
from werkzeug.utils import secure_filename

def run(video_file):
    video_file.save(os.path.join("./static/videos", secure_filename(video_file.filename)))
    video = "./static/videos" + "/" + secure_filename(video_file.filename)
    video_path = video
    video_file_name = secure_filename(video_file.filename)[:-4]

    return video, video_path, video_file_name