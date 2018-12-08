import sys
import unicodedata
import emoji
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

lmtzr = WordNetLemmatizer()
lmtzr.lemmatize("INIT") # preload
nltk.pos_tag(["INIT"]) # preload

import ProcessorChain
from util import constants


class Sanitize(ProcessorChain.BaseProcessor):
    def run(self, item):
        output = item

        output = output.strip( )
        output = output.decode('utf-8')
        output = output.lower( )

        return output, False 


class Lemmatize(ProcessorChain.BaseProcessor):
    def __init__(self, config):
        self.stops = set(stopwords.words("english") + ['etc'])
        self.table = dict.fromkeys(i for i in xrange(sys.maxunicode) if unicodedata.category(unichr(i)).startswith('P'))

    def get_wordnet_pos(self,treebank_tag):
        """
        return WORDNET POS compliance to WORDENT lemmatization (a,n,r,v) 
        """
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            # As default pos in lemmatization is Noun
            return wordnet.NOUN

    def run(self, item):
        line = item
        line = line.translate(self.table)
        words = nltk.word_tokenize(line)

        lemmas = [ ]
        for word_tag in nltk.pos_tag(words):
            lemma = lmtzr.lemmatize(word_tag[0], self.get_wordnet_pos(word_tag[1]))
            if len(lemma) >= constants.WORD_LENGTH_THRESHOLD and lemma not in self.stops and not lemma.isdigit():
                lemmas.append(lemma)

        output = ' '.join(lemmas)

        return output, False 


class Demojize(ProcessorChain.BaseProcessor):
    def run(self, item):
        output = emoji.demojize(item)

        return output, False







