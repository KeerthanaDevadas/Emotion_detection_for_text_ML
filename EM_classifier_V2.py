import gensim
from keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.layers import Dense, Activation,Dropout
from keras.optimizers import SGD
from gensim.models.word2vec import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from gensim.matutils import unitvec
import time
start = time.time()
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
w2v_file_eng="GoogleNews-vectors-negative300.bin"
w2v_file_kan='wiki.kn/wiki.kn.vec'
#w2v = gensim.models.KeyedVectors.load_word2vec_format(w2v_file_eng, binary=True)
#w2v = gensim.models.Word2Vec.load("GoogleNews-vectors-negative300.w2v")
#w2v=Word2Vec.load("20news_group.w2v")
w2v = gensim.models.KeyedVectors.load_word2vec_format(w2v_file_eng, binary=True)
print("w2v loaded successfuly..!")
def get_w2v_sentence_vector( sent_words):
    sent_words = [word for word in sent_words if word in w2v.vocab]
    if len(sent_words) > 0:
        try:
            avg_vector = np.divide(np.sum([w2v[word] for word in sent_words ], axis=0), len(sent_words))
            return unitvec(avg_vector)
        except:
            return np.zeros(300)
    else:
        return np.zeros(300)

def create_vectors(issues, vec):
    all_w2v_vectors = []
    for issue in issues:
        all_w2v_vectors.append(np.array(get_w2v_sentence_vector(issue.split())))

    count_vectors = vec.fit_transform(issues).toarray()
    count_vectors = np.array(count_vectors.tolist())

    final_vectors_list = []
    for w_vec, c_vec in zip(all_w2v_vectors, count_vectors):
        combine = []
        combine.extend(w_vec)
        combine.extend(c_vec.astype(float))
        final_vectors_list.append(combine)

    final_vectors_list = np.array(final_vectors_list)

    return final_vectors_list
def w2v_vec(issues):
    all_w2v_vectors = []
    for issue in issues:
        all_w2v_vectors.append(np.array(get_w2v_sentence_vector(issue.split())))

    return np.array(all_w2v_vectors)

def train_test_split_self(X,y,n_classes):
    #This part is currently not being used
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
X=w2v_vec(df[1].tolist())

#count_vec = CountVectorizer(stop_words='english')
#X = create_vectors(df[1].tolist(), count_vec)

print("vector creation completed")
Y=[y for y in df.loc[:,0]]
random_seed=100
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
Y =np_utils.to_categorical(encoded_Y)

X=np.array((X))




np.random.seed(random_seed)
random_indices=np.random.permutation(range(len(Y)))
split_range=int(len(random_indices)*80/100)
test_indices = random_indices[:split_range]
train_indices=random_indices[split_range:]

X_train=np.array([X[i] for i in train_indices])
X_test=np.array([X[i] for i in test_indices])
Y_train=np.array([Y[i] for i in train_indices])
Y_test=[Y[i] for i in test_indices]



model=Sequential()
model.add(Dense(300, activation='relu', input_dim=300))
model.add(Dropout(0.5))

model.add(Dense(8, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
model.fit(X_train,Y_train, epochs=60, batch_size=32)
#score = model.evaluate(np.array(X_test), np.array(Y_test), batch_size=128)
y_pred=model.predict_classes(X_test)

print('  '.join(encoder.inverse_transform(y_pred)))
print(Y_test)
#print('  '.join(encoder.inverse_transform(Y_test)))
#print(encoder.inverse_transform(Y_test))

print('It took', time.time()-start, 'seconds.')




