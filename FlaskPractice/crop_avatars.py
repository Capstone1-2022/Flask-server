import os
import torch

import utils
import detect
import model

from PIL import Image, ImageDraw, ImageFont

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run():
    checkpoint = 'BEST_checkpoint_ssd300.pth.tar'
    checkpoint = torch.load(checkpoint, map_location=device)
    start_epoch = checkpoint['epoch'] + 1
    best_loss = checkpoint['loss']
    print('\nLoaded checkpoint from epoch %d. Best loss so far is %.3f.\n' % (start_epoch, best_loss))

    model = checkpoint['model']
    model = model.to(device)
    model.eval()
    num=0

    for i in range(6):
        img_path = './static/imgs/RobloxVideo/' + str(i+1) + '.jpg'
        original_image = Image.open(img_path, mode='r')
        original_image = original_image.convert('RGB')
        try:
            os.mkdir("./static/detection/")
        except OSError:
            None
        detect_path="./static/detection/" + str(num) + ".jpg"
        num+=1
        detect.detect(original_image, min_score=0.2, max_overlap=0.1, top_k=1000, model=model).save(detect_path, "JPEG")