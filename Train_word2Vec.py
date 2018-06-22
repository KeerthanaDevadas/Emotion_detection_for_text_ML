import gensim
import matplotlib.pyplot as plt
import pandas as pd
from gensim.models.word2vec import Word2Vec
import re
from nltk.tokenize import TweetTokenizer
#import senttry
tokenizer = TweetTokenizer()

def remove_taggedname_hashtags_and_links(text):
    text1=tokenizer.tokenize(text)
    words= [i for i in text1 if not (i.startswith("@") or i.startswith("http") or i.startswith("ftp") or i.startswith("#")) ]
    sentence = ""
    for word in words:
        sentence += word + " "
    return sentence


df=pd.read_csv("train.csv",header=None,encoding='iso-8859-1')
print("data file loading completed")
raw_tweets=df[2].tolist()

tweets = []
for tweet in raw_tweets:
    clean_tweet = re.sub(r"[,:.;@#?!&$]+\ *", " ", tweet, flags=re.VERBOSE)
    tweets.append(remove_taggedname_hashtags_and_links(tweet))
    # print(tweets)
fname="Twitter.bin"
model = Word2Vec(tweets, size=200, window=5, min_count=5, workers=4)
model.save(fname)
model = Word2Vec.load(fname)