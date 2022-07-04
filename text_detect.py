import os
import glob
import cv2
import numpy as np
from os import path
from aip import AipOcr
from PIL import Image

def check(s1,s2):
    num1 = len(s1)
    num2 = len(s2)
    dp = np.zeros((num1 + 2, num2 + 2), dtype = int)

    for i in range(0, num1 + 1):
        dp[i][0] = i
    for i in range(0, num2 + 1):
        dp[0][i] = i
    for i in range(1, num1 + 1):
        for j in range(1, num2 + 1):
            if(s1[i-1] == s2[j-1]):
                dp[i][j] = dp[i-1][j-1]
            else:
                MIN = dp[i][j-1]+1
                MIN = min(MIN,dp[i-1][j]+1)
                MIN = min(MIN,dp[i-1][j-1]+1)
                dp[i][j] = MIN

    return dp[num1][num2]


def baiduOCR(picfile):
    filename = path.basename(picfile)

    # Baidu-OCR API 구독 만료
    # 구독시 ID/KEY/Secret-Key 가 발급됩니다.
    # 현재 구독 만료된 상태로 사용이 불가합니다.
    APP_ID = '-'
    API_KEY = '-'
    SECRECT_KEY = '-'
    client = AipOcr(APP_ID, API_KEY, SECRECT_KEY)

    i = open(picfile, 'rb')
    img = i.read()
    message = client.general(img)
    i.close()
    return message


def run():
    limit = 0
    m = {}
    path1 = "./static/detection/nickname"  # folder directory
    path2 = "./static/detection/nickname/box"
    files = os.listdir(path1)  # Get the names of all files in a folder
    s = []
    for file in files:  # Traverse folders
        if ".jpg" in file:  # Determine whether it is a folder, not a folder to open
            # print(file)
            message = baiduOCR(path1 + "/" + file)
            print(message)
            img = cv2.imread(path1 + "/" + file)
            num = 0
            for k in range(0, len(message['words_result'])):
                cv2.rectangle(img, (message['words_result'][k]['location']['left'], message['words_result'][k]['location']['top']), (message['words_result'][k]['location']['left']+message['words_result'][k]['location']['width'], message['words_result'][k]['location']['top']+message['words_result'][k]['location']['height']), (0, 255, 0), 2)
                cv2.imwrite(path2 + "/" + file, img)
                word = message['words_result'][0]['words']
                if(word in m):
                    m[word] += 1
                else:
                    flag = False
                    for i in m.items():
                        if(check(word,i[0]) <= 5 and len(word) > 5):
                            flag = True
                            m[i[0]] += 1
                            break
                    if(flag == False):
                        m[word] = 1

    text_count = 0
    f = open("nickname.txt",'w')
    for i in m.items():
        print(i[0])
        if(len(i[0]) > 2 and i[1] > 2):
            text_count += 1
            loop = i[1]
            if (loop > 8):
                loop = 8
            for j in range(loop):
                f.write(i[0] + '\r')
    
    return text_count
