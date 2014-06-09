#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Exposes the summarizer and its capabilities.
"""

import logging
from similaritymeasures import cosine, w2v
from gensim.models import word2vec

logger = logging.getLogger(__name__)


class Summarizer():
    """
    The summarizer. Given a set of documents,
    it will be able to generate a summary
    through sentence fusion and ILP selection.
    """

    def __init__(self, minbigramcount=4,
                 similaritymeasure=cosine, wordmodelfile=None):
        """
        Creates the summarizer.
        """

        self.minbigramcount = minbigramcount

        if similaritymeasure == w2v:
            if wordmodelfile is None:
                raise AttributeError(
                    "A word model file must "
                    "be used for w2v similarity")
            logger.info("Initializing the word model")
            self.wordmodel = word2vec.Word2Vec.load(wordmodelfile)

        self.similaritymeasure = similaritymeasure

        self.documents = None
        self.bigrams = None

    def setDocuments(self, doclist):
        """
        Set a list of document in the summarizer
        """
        if self.documents is None:
            self.documents = []
        for doc in doclist:
            self.documents.append(doc)

    def addDocument(self, doc):
        """
        Add one document in the summarizer
        """
        if self.documents is None:
            self.documents = []
        self.documents.append(doc)

    def clearDocuments(self):
        """
        Remove all documents in the summarizer
        """
        self.documents = None

    def summarize(self, wordlimit=100):
        """
        Summarize the documents into one summary
        """
        pass