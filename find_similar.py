import gensim
import numpy as np
w2v_file_eng="GoogleNews-vectors-negative300.bin"
w2v_file_kan='wiki.kn/wiki.kn.vec'
#w2v = gensim.models.KeyedVectors.load_word2vec_format(w2v_file_kan, binary=True)
#w2v = gensim.models.KeyedVectors.load_word2vec_format(w2v_file_kan, binary=False)
w2v = gensim.models.KeyedVectors.load_word2vec_format(w2v_file_eng, binary=True)
#w2v = gensim.models.Word2Vec.load("GoogleNews-vectors-negative300.w2v")
def get_w2v_sentence_vector( sent_words):
    sent_words = [word for word in sent_words if word in w2v.vocab]
    if len(sent_words) > 0:

            avg_vector = np.divide(np.sum([w2v[word] for word in sent_words], axis=0), len(sent_words))
            return avg_vector
    return None


def calculate_class(sentence):

    tag1 = w2v["sad"]
    tag2 = w2v["happy"]

    sentence_vec=get_w2v_sentence_vector(sentence.split())
    #sad_score = w2v.similarity(sentence_vec, tag1)
    #happy_score = w2v.similarity(sentence_vec, tag2)
    score=w2v.cosine_similarities(sentence_vec,[tag1,tag2])
    print(score)
    if(score[0]>score[1]):
        print("sentence: {0:s} :: belongs to {1:s} class" .format(sentence,"sad"))

    else:
        print("sentence: {0:s} belongs to {1:s} class".format(sentence, "happy"))


#sentence = "very bad political move"
#calculate_class(sentence)

# Load Google's pre-trained Word2Vec model.

while(True):
    test_word=input("enter the word :")
    print(w2v.most_similar(test_word,topn=20))

#print(w2v.most_si  milar("ಸಾಹಿತ್ಯ",topn=10))

#sentence= input("enter the sentence to be evaluated")








# sentence_vec=get_w2v_sentence_vector((sentence.split()))
#
#
# topn = 10;
# most_similar_words = w2v.most_similar( [ sentence_vec],[], topn)
#
# print(most_similar_words)







