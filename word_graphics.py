import gensim
import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot
from gensim.models import word2vec
import nltk
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

w2v_file_eng="GoogleNews-vectors-negative300.bin"
w2v_file_kan='wiki.kn/wiki.kn.vec'
words_graph_file="words_graph.pdf"
w2v = gensim.models.KeyedVectors.load_word2vec_format(w2v_file_eng, binary=True)
'''
x=w2v[w2v.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(x)

fig=pyplot.scatter(result[:10, 0], result[:10, 1])

words = list(w2v.vocab)
for i, word in enumerate(words):
	pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))

fig.savefig(words_graph_file)
pyplot.show()
'''
STOP_WORDS = nltk.corpus.stopwords.words()



def tsne_plot(w2v):

	"Creates and TSNE model and plots it"
	labels = []
	tokens = []
	k=0
	'''
	for word in model.vocab:

		if word not in STOP_WORDS:
			tokens.append(model[word])
			labels.append(word)
			k += 1
			if (k == 30):
				break
	'''
	words=["king","queen","book","novel","biography","car","bike"]
	for word in words:
		tokens.append(w2v[word])
		labels.append(word)





	tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
	new_values = tsne_model.fit_transform(tokens)

	x = []
	y = []
	for value in new_values:
		x.append(value[0])
		y.append(value[1])

	plt.figure(figsize=(16, 16))
	for i in range(len(x)):
		plt.scatter(x[i], y[i])
		plt.annotate(labels[i],
					 xy=(x[i], y[i]),
					 xytext=(5, 2),
					 textcoords='offset points',
					 ha='right',
					 va='bottom')
	print(w2v.most_similar("car", topn=20))
	plt.show()
	print('hello')
#model = word2vec.Word2Vec(corpus, size=100, window=20, min_count=500, workers=4)
tsne_plot(w2v)