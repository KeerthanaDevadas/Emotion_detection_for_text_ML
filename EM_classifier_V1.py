import gensim
import matplotlib.pyplot as plt
import re
import csv
from nltk import tokenize
from gensim.models.word2vec import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from gensim.matutils import unitvec
from sklearn.externals import joblib
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
import pytypo
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")
tokenizer = TweetTokenizer()

from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import time
start = time.time()
print("started")
w2v_file_eng="GoogleNews-vectors-negative300.bin"
#w2v_file_kan='wiki.kn/wiki.kn.vec'
#w2v_file_eng="glove.twitter.27B.100d.txt"
# w2v = gensim.models.KeyedVectors.load_word2vec_format(w2v_file_eng, binary=True)
#w2v = gensim.models.Word2Vec.load("GoogleNews-vectors-negative300.w2v")
#w2v=Word2Vec.load("Twitter.bin")
#w2v=gensim.models.KeyedVectors.load_word2vec_format("word2vec_twitter_model.bin",binary=True,unicode_errors="ignore")
w2v = gensim.models.KeyedVectors.load_word2vec_format(w2v_file_eng, binary=True)

print("w2v loaded successfuly..!")
def get_w2v_sentence_vector( sent_words):
    '''return the average vector for a sentence by calculating average over vectors of
    each word in the sentence'''
    sent_words = [word for word in sent_words if word in w2v.vocab]
    if len(sent_words) > 0:
        try:
            avg_vector = np.divide(np.sum([w2v[word] for word in sent_words ], axis=0), len(sent_words))
            return unitvec(avg_vector)
        except:
            return np.zeros(300)
    else:
        return np.zeros(300)

def create_vectors(issues, vec,fit):
    '''return the combined vector odf word2vec and count vectorizer.
    '''
    all_w2v_vectors = []
    for issue in issues:
        all_w2v_vectors.append(np.array(get_w2v_sentence_vector(issue.split())))
    if(fit==True):
        count_vectors = vec.fit_transform(issues).toarray()
    else:
        count_vectors=vec.transform(issues).toarray()
    count_vectors = np.array(count_vectors.tolist())

    final_vectors_list = []
    for w_vec, c_vec in zip(all_w2v_vectors, count_vectors):
        combine = []
        combine.extend(w_vec)
        combine.extend(c_vec.astype(float))
        final_vectors_list.append(combine)

    final_vectors_list = np.array(final_vectors_list)

    return pd.DataFrame(final_vectors_list)

def plot_class_score(precision,recall, threshold, average_precision):
    '''
    Plot the class wise pr curve

    '''

    plt.figure()

    for i in range(len(clf.classes_)):
        plt.plot(recall[i], precision[i],
                 label='Precision-recall curve of class {0} (area = {1:0.2f})'
                       ''.format(clf.classes_[i], average_precision[i]))

    plt.plot(recall["micro"], precision["micro"],
             label='micro-average Precision-recall curve (area = {0:0.2f})'
                   ''.format(average_precision["micro"]))

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc="lower right")
    plt.savefig("pr_class.png")
    plt.savefig("pr_class.pdf")
def plot_avg(precision,recall, threshold, average_precision):
    '''
    plot the average precision- recall curve
    '''
    plt.figure()
    plt.plot(np.append(threshold['micro'],[1]), recall['micro'],'g-',label='average recall')
    plt.plot(np.append(threshold['micro'],[1]), precision['micro'],'r-', label='average precision')


    plt.xlabel('Threshold')
    plt.ylabel('Recall-Precision')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc="lower right")
    plt.savefig("prt_avg.png")
    plt.savefig("prt_avg.pdf")
def precision_recall(y_test,y_score,n_classes):
    precision = dict()
    recall = dict()
    average_precision = dict()
    threshold = dict()
    y_test2 = []
    y_score2 = []
    for i in range(0, n_classes):
        y_tst = [1 if y == i else 0 for y in y_test]
        y_scr = [scr[i] for scr in y_score]
        y_score[i]
        y_test2.append(y_tst)
        y_score2.append(y_scr)
        precision[i], recall[i], threshold[i] = precision_recall_curve(y_tst,
                                                                       y_scr)
        average_precision[i] = average_precision_score(y_tst, y_scr)

    precision["micro"], recall["micro"], threshold['micro'] = precision_recall_curve(np.array(y_test2).ravel(),
                                                                                     np.array(y_score2).ravel())
    average_precision["micro"] = average_precision_score(np.array(y_test2), np.array(y_score2),
                                                         average="micro")
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'
          .format(average_precision["micro"]))

    return precision,recall,threshold,average_precision

def remove_taggedname_hashtags_and_links(text):
    text1=tokenizer.tokenize(text)
    words= [i for i in text1 if not (i.startswith("@") or i.startswith("http") or i.startswith("ftp")) ]
    sentence = ""
    for word in words:
        if(word.startswith("#")):
            sentence+=word.substring(1)
        else:
            sentence += word + " "
    return sentence


def w2v_vec(issues):
    all_w2v_vectors = []
    for issue in issues:
        all_w2v_vectors.append(np.array(get_w2v_sentence_vector(issue.split())))

    return pd.DataFrame(all_w2v_vectors)

def send_vec(tweets):
    '''when handling real time tweets.Can be replaced by w2v_vec() as well'''
    combine=[]
    for tweet in tweets:
        vec=np.array(get_w2v_sentence_vector(tweet))
    combine.extend(vec)
    xv=np.array(combine).reshape(1,-1)
    return xv

def train_test_split_self(X,y,n_classes):
    #This function is currently not being used
    classes = {}
    for ind,val in enumerate(y):
        if val not in classes:
            classes[val] = ind
        if n_classes == len(classes):
            break
    X_filtered = [val for ind,val in enumerate(X) if ind not in classes.values()]
    y_filtered = [val for ind, val in enumerate(y) if ind not in classes.values()]

    X_train, X_test, y_train, y_test =train_test_split(X_filtered,y_filtered,test_size=0.2,random_state=0)
    for val in classes.values():
        X_train.append(X[val])
        y_train.append(y[val])

    return X_train, X_test, y_train, y_test

df=pd.read_csv("tagged-1.csv",header=None,encoding='iso-8859-1')
print("data file loading completed")
#X=[get_w2v_sentence_vector(x) for x in df.loc[:,1]]
list_of_tweets=[]
for line in df[1].tolist():
    try:
        list_of_tweets.append(pytypo.correct_sentence(line))
    except:
        list_of_tweets.append(line)

X=w2v_vec(list_of_tweets)


# count_vec = CountVectorizer(stop_words='english')
# X = create_vectors(list_of_tweets, count_vec,fit=True)

mlp_classifier_filename = "mlp_clf_w2v.pkl"

print("vector creation completed")
Y=[y for y in df.loc[:,0]]
random_seed=100


np.random.seed(random_seed)
random_indices=np.random.permutation(range(len(Y)))
split_range=int(len(random_indices)*80/100)
test_indices = random_indices[:split_range]
train_indices=random_indices[split_range:]

X_train=[X.loc[i,:] for i in train_indices]
X_test=[X.loc[i,:] for i in test_indices]
Y_train=[Y[i] for i in train_indices]
Y_test=[Y[i] for i in test_indices]




if False:
    clf = MLPClassifier(solver="adam", hidden_layer_sizes=100, alpha=0.01, activation='logistic')
    clf.fit(X_train, Y_train)
    mlp_classifier_file = open(mlp_classifier_filename, 'wb')
    pickle.dump(clf, mlp_classifier_file)
    mlp_classifier_file.close()

else:
    mlp_classifier_file = open(mlp_classifier_filename, 'rb')
    clf = pickle.load(mlp_classifier_file)





Y_pred=clf.predict(X_test)
print('  '.join(Y_pred))
print(Y_test)
accuracy=accuracy_score(Y_test, Y_pred)
prob=clf.predict_proba(X_test)
print(classification_report(Y_test, Y_pred))
print(accuracy)


y_score = clf.predict_proba(X_test)

y_dict=dict(zip(clf.classes_,range(6)))

y_test_num=[y_dict[i] for i in Y_test]

precision, recall, threshold, average_precision = precision_recall(y_test_num, y_score, n_classes=6)
plot_class_score(precision, recall, threshold, average_precision)
plot_avg(precision, recall, threshold, average_precision)



print('It took', time.time()-start, 'seconds.')

def test_real_data():
    try:
        import senttry
        key = e1.get()
        raw_tweets = senttry.get_tweets(key)[:10]
        print(raw_tweets)
        tweets = []
        for tweet in raw_tweets:
            clean_tweet = re.sub(r"[,:!.;@#?!&$]+\ *", " ", tweet, flags=re.VERBOSE)
            tweets.append(remove_taggedname_hashtags_and_links(clean_tweet))

        data_input = w2v_vec(tweets)
        #data_input=create_vectors(tweets,count_vec,fit=False)
        data_pred = clf.predict(data_input)
        print(data_pred)

        t.delete(1.0, END)
        t.tag_config('label', foreground="red")
        for i in range(len(data_pred)):
            # Label(master, text=raw_tweets[i] + "----->" + data_pred[i], justify=tk.LEFT).grid(row=7 + 2 * i, column=0,
            #                                                                                   sticky=W, pady=7)


            t.insert(END, raw_tweets[i]+"----->")
            t.insert(END,data_pred[i] +'\n\n',"label")
            t.pack(fill=BOTH)
        with open("emotion_predicted.csv", "w", encoding="utf-8") as res:
            writer = csv.writer(res)
            for i in range(len(tweets)):
                writer.writerow([tweets[i], data_pred[i]])
        res.close()




    except:
        print("check your connection.. No internet or connection is restricting twitter api.")
        t.delete(1.0, END)
        t.insert(END,"check your connection.. No internet or connection is restricting twitter api." )
        t.pack(fill=BOTH)







from tkinter import *
import tkinter as tk
master = Tk()
Label(master, text="Enter the hashtag:").pack()
master.geometry("1500x1500")
e1 = Entry(master)
e1.insert(0,"#")
e1.pack()

t = Text(master)
t.insert(END, "your tweets will appear here"+'\n')
t.pack()
Button(master, text='Quit', command=master.quit).pack()
Button(master, text='Enter', command=test_real_data).pack()
mainloop()















