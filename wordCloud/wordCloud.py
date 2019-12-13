from wordcloud import WordCloud
import jieba
import re

f=open("test.txt","r")#txt's name
txt=f.read()
text=txt.strip()
r='[， 。 、 ； 1234567890]'#add the word which you needn't
text=re.sub(r,'',text)
text= jieba.lcut(text)
text=" ".join(text)
wordcloud=WordCloud(font_path="simkai.ttf",background_color="white",width=1200,height=660).generate(text)
wordcloud.to_file("pic.png")