import os

for file in os.scandir("./static/imgs"):
    os.remove(file.path)

for file in os.scandir("./static/detection"):
    if(os.path.isfile(file.path)):
        os.remove(file.path)

for file in os.scandir("./static/detection/crop"):
    os.remove(file.path)