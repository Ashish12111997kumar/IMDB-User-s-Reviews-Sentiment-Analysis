import requests
from bs4 import BeautifulSoup as bs
import re
import nltk
from PIL import Image
from os import path, getcwd
import numpy as np
nltk.download('stopwords')
from nltk.corpus import stopwords
from wordcloud import WordCloud,ImageColorGenerator
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

extract=requests.get('https://www.imdb.com/title/tt10295212/reviews')
soup=bs(extract.content,'html.parser')
extract=soup('div',class_='text show-more__control')
reviews=[]
for i in range(len(extract)):
  reviews.append(extract[i].text)
reviews=' '.join(reviews)
reviews=re.sub("[^a-zA-Z" "]+"," ",reviews).lower()
review_split=reviews.split(" ")
tfidf=TfidfVectorizer(review_split,use_idf=True,ngram_range=(1,2),stop_words=stopwords.words('english'))
tfidf_matrix=tfidf.fit_transform(review_split)
#d=getcwd()
#mask=np.array(Image.open(path.join(d,'/content/drive/MyDrive/parents.png')))
wc=WordCloud(width=1400, height=1800,background_color='white',max_words=150,mask=mask).generate(reviews)
#image_colors = ImageColorGenerator(mask)
#plt.figure(figsize=[12,12])
#plt.imshow(wc.recolor(color_func=image_colors), interpolation="bilinear")
#plt.axis("off")
plt.imshow(wc)
# positve words
with open('/content/drive/MyDrive/positive-words.txt') as df:
  pos=df.read()
pos=pos.split('\n')
positive=' '.join([i for i in review_split if i in pos])
#d=getcwd()
#mask=np.array(Image.open(path.join(d,'/content/drive/MyDrive/parents.png')))
wc1=WordCloud(width=1400, height=1800,background_color='white',max_words=150).generate(positive)
#img_C=ImageColorGenerator(mask)
#plt.figure(figsize=(15,15))
#plt.imshow(wc1.recolor(color_func=img_C),interpolation='bilinear')
#plt.axis("off")
plt.imshow(wc1)

# negative words

with open('/content/drive/MyDrive/negative-words.txt') as df:
  neg=df.read()
neg=neg.split('\n')
negative=' '.join([i for i in review_split if i in neg])
#d=getcwd()
#mask=np.array(Image.open(path.join(d,'/content/drive/MyDrive/dore.jpg')))
wc2=WordCloud(width=1400, height=1800,background_color='white',max_words=150).generate(negative)
#img_C=ImageColorGenerator(mask)
#plt.figure(figsize=(15,15))
#plt.imshow(wc1.recolor(color_func=img_C),interpolation='bilinear')
#plt.axis("off")
plt.imshow(wc2)

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('wordnet')
reviews=re.sub(r'[^a-zA-Z]',' ',reviews)
tokens=word_tokenize(reviews)
text1=nltk.Text(tokens)
lem=WordNetLemmatizer()
text_content = [''.join(re.split("[ .,;:!?‘’``''@#$%^_&*()<>{}~\n\t\\\-]", word)) for word in text1]
text_content=[i for i in text_content if i not in set(stopwords.words('english'))]
text_content=[i for i in text_content if len(i)!=0]
text_content=[lem.lemmatize(i) for i in text_content]
bigrm=list(nltk.bigrams(text_content))
big_st=[' '.join(i) for i in bigrm]
from sklearn.feature_extraction.text import CountVectorizer
cnt=CountVectorizer(ngram_range=(2,3))
bow=cnt.fit_transform(big_st)
sum_words = bow.sum(axis=0)
word_freq=[(word,sum_words[0,idx]) for word,idx in cnt.vocabulary_.items()]
word_freq=word_freq=sorted(word_freq,key=lambda x:x[1],reverse=True)
words_dict = dict(word_freq)
wc3=WordCloud(width=1400, height=2400,max_words=190,background_color='white')
plt.figure(figsize=(10,10))
wc3.generate_from_frequencies(words_dict)
#plt.axis('off')
plt.imshow(wc3)
