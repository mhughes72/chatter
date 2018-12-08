import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from util import constants


def sent2vec(sentences, vocab):
    vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             vocabulary = vocab, \
                             max_features = constants.MAX_FEATURES) 

    vectors = vectorizer.fit_transform(sentences)

    return vectors.toarray().astype('float32')