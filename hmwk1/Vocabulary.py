from collections import Counter 
from re import sub, compile
import matplotlib.pyplot as plt
import numpy as np

class UnimplementedFunctionError(Exception):
	pass

class Vocabulary:

	def __init__(self, corpus):

		self.word2idx, self.idx2word, self.freq = self.build_vocab(corpus)
		self.size = len(self.word2idx)

	def most_common(self, k):
		freq = sorted(self.freq.items(), key=lambda x: x[1], reverse=True)
		return [t for t,f in freq[:k]]


	def text2idx(self, text):
		tokens = self.tokenize(text)
		return [self.word2idx[t] if t in self.word2idx.keys() else self.word2idx['UNK'] for t in tokens]

	def idx2text(self, idxs):
		return [self.idx2word[i] if i in self.idx2word.keys() else 'UNK' for i in idxs]


	###########################
	## TASK 1.1           	 ##
	###########################
	def tokenize(self, text):
		"""
	    
	    tokenize takes in a string of text and returns an array of strings splitting the text into discrete tokens.

	    :params: 
	    - text: a string to be tokenize, e.g. "The blue dog jumped, but not high."

	    :returns:
	    - tokens: a list of strings derived from the text, e.g. ["the", "blue", "dog", "jumped", "but", "not", "high"] for word-level tokenization
	    
	    """ 
		finder = compile('[a-zA-Z]+')
		return finder.findall(text)



	###########################
	## TASK 1.2            	 ##
	###########################
	def build_vocab(self,corpus):
		"""
	    
	    build_vocab takes in list of strings corresponding to a text corpus, tokenizes the strings, and builds a finite vocabulary

	    :params:
	    - corpus: a list string to build a vocabulary over

	    :returns: 
	    - word2idx: a dictionary mapping token strings to their numerical index in the dictionary e.g. { "dog": 0, "but":1, ..., "UNK":129}
	    - idx2word: the inverse of word2idx mapping an index in the vocabulary to its word e.g. {0: "dog", 1:"but", ..., 129:"UNK"}
	    - freq: a dictionary of words and frequency counts over the corpus (including words not in the dictionary), e.g. {"dog":102, "the": 18023, ...}

	    """ 
		self.thresh = 50
		freq = {}
		for sent in corpus:
			for wrd in self.tokenize(sent):
				try:
					freq[wrd.lower()] += 1
				except KeyError:
					freq[wrd.lower()] = 1

		srted = sorted(freq.items(), key=lambda item: item[1], reverse=True)
		word2idx = {}
		idx2wrd = {}
		idx = 0
		for vals in srted:			
			if vals[1] > self.thresh:
				word2idx[vals[0]] = idx
				idx2wrd[idx] = vals[0]
				idx += 1
				self.cutoff = idx
			else:
				word2idx["UNK"] = idx
				idx2wrd[idx] = "UNK"

		return word2idx, idx2wrd, freq


	###########################
	## TASK 1.3              ##
	###########################
	def make_vocab_charts(self):
		"""
	    
	    make_vocab_charts plots word frequency and cumulative coverage charts for this vocabulary. See handout for more details

	    
	    """ 
		fig, axs = plt.subplots(1, 2)
		# plot 1
		com = sorted(self.freq.values(), reverse=True)
		axs[0].set_yscale('log')
		axs[0].plot(com)
		axs[0].axhline(self.thresh, color='red')
		axs[0].set_title('Token Frequency Distribution')
		axs[0].set_ylabel('Frequency')
		axs[0].set_xlabel('Token ID by Frequency')

		# plot 2
		plt.yscale('linear')
		y = np.cumsum(com)
		y = y / y[-1]
		axs[1].plot(y)
		print(y[self.cutoff])
		axs[1].axvline(self.cutoff, color='red')
		axs[1].set_title('Cumulative Fraction Covered')
		axs[1].set_ylabel('Fraction Covered')
		axs[1].set_xlabel('Token ID by Frequency')

		plt.show()