#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Exposes the summarizer and its capabilities.
"""

import logging
from similaritymeasures import cosine, w2v
from gensim.models import word2vec
from collections import Counter

logger = logging.getLogger(__name__)


def _findBestMatch(matrix):
    """
    Finds the two most similar items in a similarity matrix.
    Returns the top score and the matching pair indexes, ordered.
    """
    bestscore = 0
    pairtomerge = None
    for rnum, row in enumerate(matrix[1:]):
        for cnum, score in enumerate(row[1:]):
            if score > bestscore and cnum != rnum:
                bestscore = score
                pairtomerge = (rnum+1, cnum+1)
    if pairtomerge:
        rnum = min(pairtomerge)
        cnum = max(pairtomerge)
    return bestscore, (rnum, cnum)


def _mergeClusters(matrix, threshold):
    """
    Merge clusters given a similarity matrix and a threshold
    """
    # find max value in matrix
    bestscore, (rnum, cnum) = _findBestMatch(matrix)
    finished = bestscore < threshold

    while not finished:
        # merge sentences in one cluster
        matrix[0][rnum] = matrix[0][rnum] + matrix[0][cnum]
        matrix[rnum][0] = matrix[rnum][0] + matrix[cnum][0]

        # update values
        for cind, score in enumerate(matrix[rnum][1:]):
            matrix[rnum][cind+1] = min(matrix[cnum][cind+1], score)
        for rind, row in enumerate(matrix[1:]):
            if rind + 1 > cnum:
                row[rnum] = min(row[rnum], matrix[rind+1][cnum])

        # remove remaining entries
        for row in matrix:
            if len(row) > cnum:
                row.pop(cnum)
        matrix.pop(cnum)

        bestscore, (rnum, cnum) = _findBestMatch(matrix)
        finished = bestscore < threshold
    return matrix[0][1:]


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
        self.bigramstats = None

    def setDocuments(self, doclist):
        """
        Set a list of document in the summarizer
        """
        for doc in doclist:
            self.addDocument(doc)

    def addDocument(self, doc):
        """
        Add one document in the summarizer
        """
        if self.documents is None:
            self.documents = []
        self.documents.append(doc)
        self.updateBigrams(doc)

    def clearDocuments(self):
        """
        Remove all documents in the summarizer
        """
        self.documents = None
        self.bigramstats = None

    def updateBigrams(self, doc):
        """
        Update the bigram statistics for the summarizer
        """
        if self.bigramstats is None:
            self.bigramstats = Counter()

        # check that we consider bigrams only
        # once per document (document frequency)
        seen = set()
        for sentence in doc.bigrams:
            for bigram in sentence:
                if bigram not in seen:
                    self.bigramstats[bigram] += 1
                    seen.add(bigram)

    def _clusterSentences(self, sim):
        """
        Clusters the documents' sentences into related
        clusters given a similarity matrix sim
        """
        # get stemmed sentences from all documents
        sentences = [sentence for doc in self.documents
                     for sentence in doc.stemTokens]
        # get full sentences for clean clusters
        fullsentences = [sentence for doc in self.documents
                         for sentence in doc.sentences]
        # stemmed sentences to string
        strsentences = [" ".join(['/'.join(token)
                        for token in sentence])
                        for sentence in sentences]

        # computes triangular similarity matrix
        matrix = [[sim(s1, s2)
                   for i1, s1 in enumerate(strsentences)
                   if i1 <= i2]
                  for i2, s2 in enumerate(strsentences)]

        # add sentences on first row and first column
        for i in range(len(matrix)):
            matrix[i].insert(0, [fullsentences[i]])
        matrix.insert(0, [None] + [[sentence] for sentence in fullsentences])

        # gets the sentence clusters
        self.clusters = _mergeClusters(matrix, 0.5)

    def summarize(self, wordlimit=100):
        """
        Summarize the documents into one summary
        """
        logger.info("Clustering sentences")
        self._clusterSentences(self.similaritymeasure)


import document
s = Summarizer()
s.addDocument(document.Document("../tests/testdata/smalldoc.txt"))
s.addDocument(document.Document("../tests/testdata/smalldocb.txt"))
s.summarize()
