from wordcloud import WordCloud
from collections import Counter

def run():
    text = open('nickname.txt').read()
    text = text.replace(" ", "_")
    text = text.replace(":", "_")
    text = text.replace("+", "_")

    wc = WordCloud(max_font_size=200, font_path='arial.ttf', background_color='white', width=700, height=700)
    cloud = wc.generate(text)
    cloud.to_file('static/word_cloud/word_cloud.jpg')