import os
import torch

import utils
import cut_avatars
import model

from PIL import Image, ImageDraw, ImageFont

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run(img_num):
    checkpoint = 'object_detection/BEST_checkpoint_ssd300.pth.tar'
    checkpoint = torch.load(checkpoint, map_location=device)
    start_epoch = checkpoint['epoch'] + 1
    best_loss = checkpoint['loss']
    print('\nLoaded checkpoint from epoch %d. Best loss so far is %.3f.\n' % (start_epoch, best_loss))

    model = checkpoint['model']
    model = model.to(device)
    model.eval()
    num=0

    for i in range(img_num):
        img_path = "./static/imgs/" + str(i+1) + ".jpg"
        original_image = Image.open(img_path, mode='r')
        original_image = original_image.convert('RGB')
        try:
            os.mkdir("./static/detection/")
        except OSError:
            None
        detect_path="./static/detection/" + str(num) + ".jpg"
        num+=1
        cut_avatars.detect(original_image, min_score=0.2, max_overlap=0.1, top_k=1000, model=model, num=num).save(detect_path, "JPEG")