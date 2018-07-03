#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Preprocess documents to be summarized
"""

import nltk
import codecs
import re
import string
import logging
import os.path

logger = logging.getLogger(__name__)
# resources for postagger
_POSTAGGER = None
_currentdir = os.path.dirname(os.path.realpath(__file__))
_POSMODEL = _currentdir + '/data/stanford-en'
_POSJAR = _currentdir + '/data/stanford-postagger.jar'
_STEMMER = None


def sentTokenize(text):
    """
    Basic sentence tokenizer using nltk punkt
    """
    sentTokenizer = nltk.data.load('file:' +
                                   _currentdir +
                                   '/data/english.pickle')
    return sentTokenizer.tokenize(text)


def postag(sentence):
    """
    Postag utility using Stanford POStagger
    """
    global _POSTAGGER
    if _POSTAGGER is None:
        _POSTAGGER = nltk.tag.stanford.StanfordPOSTagger(
            _POSMODEL, _POSJAR, encoding='utf-8')

    tagsentence = _POSTAGGER.tag(sentence)

    # replace punctuation with PUNCT tag
    tagsentencepunct = []
    for tok, pos in tagsentence:
        allpunct = all(c in string.punctuation for c in tok)
        if tok in string.punctuation or allpunct:
            tagsentencepunct.append((tok, 'PUNCT'))
        else:
            tagsentencepunct.append((tok, pos))

    return tagsentencepunct


def stem(word):
    """
    Stems a word
    """
    global _STEMMER

    if _STEMMER is None:
        _STEMMER = nltk.stem.SnowballStemmer("english")

    return _STEMMER.stem(word)


def normalize(text):
    """
    Removes newlines and multiple whitespace charaters
    """
    text = text.strip()
    text = re.sub('[\n\t]', ' ', text)
    text = re.sub('\s+', ' ', text)

    return text


def isGoodToken(token, stopwords=nltk.corpus.stopwords.words('english')):
    tok, pos = token
    return tok.lower() not in stopwords and pos != 'PUNCT'


class Document():
    """
    A document. Contains different representations of the document
    that will be used for summarization.
    """

    def __init__(self, docfile, skipPreprocess=False):
        """
        Initialize a document and preprocesses it by default.
        One can use its own preprocessing method but must define
        the fields tokens, taggedTokens and stemTokens.
        """
        with codecs.open(docfile, 'r', 'utf-8') as doc:
            self.content = doc.read()
        self.docfile = docfile

        self.content = normalize(self.content)

        if not skipPreprocess:
            self.preprocess()

    def preprocess(self, sentTokenizer=sentTokenize,
                   wordTokenizer=nltk.tokenize.word_tokenize,
                   stopwords=nltk.corpus.stopwords.words('english'),
                   postagger=postag):
        """
        Preprocess the content of a document.
        """
        logger.info("Preprocessing document %s",
                    os.path.basename(self.docfile))

        self.sentences = sentTokenizer(self.content)
        self.tokens = [wordTokenizer(sentence)
                       for sentence in self.sentences]
        self.taggedTokens = [postag(toksentence)
                             for toksentence in self.tokens]
        self.filteredTokens = [[(tok, pos)
                                for tok, pos in sentence
                                if isGoodToken((tok, pos), stopwords)]
                               for sentence in self.taggedTokens]
        self.stemTokens = [[(stem(tok), pos)
                            for tok, pos in sentence]
                           for sentence in self.filteredTokens]
