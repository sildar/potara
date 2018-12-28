#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Exposes the summarizer and its capabilities.
"""

import logging
from .similaritymeasures import cosine, w2v
from gensim.models import word2vec
from collections import Counter
from .takahe import word_graph, keyphrase_reranker
import multiprocessing
from .document import stem
from pulp import GLPK
import pulp
import nltk
import string
import re

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


def _dofuse(cluster):
    """
    Extracts the call to takahe to interrupt it if it's taking too long.
    """
    fuser = word_graph(cluster, nb_words=6,
                       lang="en", punct_tag="PUNCT")
    # get fusions
    fusions = fuser.get_compression(50)
    # rerank and keep top 10
    reranker = keyphrase_reranker(cluster, fusions, lang="en")
    rerankedfusions = reranker.rerank_nbest_compressions()[0:10]
    return rerankedfusions


def _fuseCluster(cluster):
    """
    Creates alternatives to sentences in a cluster by fusing them.
    """
    # fuse only if we have 2 or more sentences
    if len(set(cluster)) < 2:
        return cluster

    # small hack. takahe module may not finish.
    # We give it 3 seconds to compute the result
    try:
        process = multiprocessing.Pool(processes=1)
        res = process.apply_async(_dofuse, (cluster,))
        rerankedfusions = res.get(timeout=3)
        process.terminate()
    except:
        # may fail if there is no verb in the cluster or illformed sentences
        rerankedfusions = []
    # recompose sentences
    finalfusions = []
    for _, fusedsentence in rerankedfusions:
        finalfusions.append(" ".join([word + '/' + pos
                                      for word, pos in fusedsentence]))
    # using sets to remove duplicates
    return list(set(finalfusions) | set(cluster))


def removePOS(sentence):
    """
    Removes the part of speech from a string sentence.
    """
    cleansentence = []
    for word in sentence.split():
        cleansentence.append(word.split('/')[0])
    return ' '.join(cleansentence)


def extractBigrams(sentence):
    """
    Extracts the bigrams from a tokenized sentence.
    Applies some filters to remove bad bigrams
    """
    bigrams = [(stem(tok1.lower()),  stem(tok2.lower()))
               for tok1, tok2 in zip(sentence, sentence[1:])]

    # filter bigrams
    bigrams = [(tok1, tok2) for tok1, tok2 in bigrams
               if not (tok1 in nltk.corpus.stopwords.words('english') and
                       tok2 in nltk.corpus.stopwords.words('english')) and
               tok1 not in string.punctuation and
               tok2 not in string.punctuation]
    return bigrams


def wellformatize(s):
    ws = re.sub(" ('[a-z]) ", "\g<1> ", s)
    ws = re.sub(" ([\.;,-]) ", "\g<1> ", ws)
    ws = re.sub(" ([\.;,-?!])$", "\g<1>", ws)
    ws = re.sub(" ' ", "' ", ws)
    ws = re.sub(" _ (.+) _ ", " -\g<1>- ", ws)
    ws = re.sub("`` ", "\"", ws)
    ws = re.sub(" ''", "\"", ws)
    ws = re.sub("\s+", " ", ws)
    # repair do n't to don't
    ws = re.sub("(\w+) n't", "\g<1>n't", ws)
    return ws.strip()


class Summarizer():
    """
    The summarizer. Given a set of documents,
    it will be able to generate a summary
    through sentence fusion and ILP selection.
    """

    def __init__(self, minbigramcount=4,
                 similaritymeasure=cosine, minsimilarity=0.3,
                 wordmodelfile=None):
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
        self.minsimilarity = minsimilarity

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
        for sentence in doc.tokens:
            for bigram in extractBigrams(sentence):
                if bigram not in seen:
                    self.bigramstats[bigram] += 1
                    seen.add(bigram)

    def clusterSentences(self):
        """
        Clusters the documents' sentences into related
        clusters given a similarity matrix sim
        """
        # get stemmed sentences from all documents
        sentences = [sentence for doc in self.documents
                     for sentence in doc.stemTokens]
        # get tagged sentences for clean clusters
        fullsentences = [sentence for doc in self.documents
                         for sentence in doc.taggedTokens]
        # pair of (word, pos) to string
        fullsentences = [" ".join(['/'.join(token)
                                   for token in sentence])
                         for sentence in fullsentences]
        # pair of (stem, pos) to string
        strsentences = [" ".join(['/'.join(token)
                        for token in sentence])
                        for sentence in sentences]

        # computes triangular similarity matrix
        matrix = [[self.similaritymeasure(s1, s2)
                   for i1, s1 in enumerate(strsentences)
                   if i1 <= i2]
                  for i2, s2 in enumerate(strsentences)]

        # add sentences on first row and first column
        for i in range(len(matrix)):
            matrix[i].insert(0, [fullsentences[i]])
        matrix.insert(0, [None] + [[sentence] for sentence in fullsentences])

        # gets the sentence clusters
        self.clusters = _mergeClusters(matrix, self.minsimilarity)

        # filters clusters
        updatedclusters = []
        for cluster in self.clusters:
            updatedcluster = []
            for sentence in cluster:
                updatedcluster.append(sentence)
            if len(updatedcluster) > 0:
                updatedclusters.append(updatedcluster)
        self.clusters = updatedclusters

    def _selectSentences(self, wordlimit):
        """
        Optimally selects the sentences based on their bigrams
        """
        fullsentences = [removePOS(sentence)
                         for cluster in self.candidates
                         for sentence in cluster]

        # remember the correspondance between a sentence and its cluster
        clusternums = {}
        sentencenums = {s: i for i, s in enumerate(fullsentences)}
        for i, cluster in enumerate(self.candidates):
            clusternums[i] = []
            for sentence in cluster:
                fullsentence = removePOS(sentence)
                if fullsentence in sentencenums:
                    clusternums[i].append(sentencenums[removePOS(sentence)])

        # extract bigrams for all sentences
        bigramssentences = [extractBigrams(sentence.split())
                            for sentence in fullsentences]

        # get uniqs bigrams
        uniqbigrams = set(bigram
                          for sentence in bigramssentences
                          for bigram in sentence)
        numbigrams = len(uniqbigrams)
        numsentences = len(fullsentences)

        # rewrite fullsentences
        fullsentences = [wellformatize(sentence) for sentence in fullsentences]

        # filter out rare bigrams
        weightedbigrams = {bigram: (count if count >= self.minbigramcount
                                    else 0)
                           for bigram, count in self.bigramstats.items()}

        problem = pulp.LpProblem("Sentence selection", pulp.LpMaximize)

        # concept variables
        concepts = pulp.LpVariable.dicts(name='c',
                                         indexs=range(numbigrams),
                                         lowBound=0,
                                         upBound=1,
                                         cat='Integer')
        sentences = pulp.LpVariable.dicts(name='s',
                                          indexs=range(numsentences),
                                          lowBound=0,
                                          upBound=1,
                                          cat='Integer')

        # objective : maximize wi * ci (weighti * concepti)
        # small hack. If the bigram has been filtered out from uniqbigrams,
        # we give it a weight of 0.
        problem += sum([(weightedbigrams.get(bigram) or 0) * concepts[i]
                        for i, bigram in enumerate(uniqbigrams)])

        # constraints

        # size
        problem += sum([sentences[j] * len(fullsentences[j].split())
                       for j in range(numsentences)]) <= wordlimit

        # integrity constraints (link between concepts and sentences)
        for j, bsentence in enumerate(bigramssentences):
            for i, bigram in enumerate(uniqbigrams):
                if bigram in bsentence:
                    problem += sentences[j] <= concepts[i]

        for i, bigram in enumerate(uniqbigrams):
            problem += sum([sentences[j]
                            for j, bsentence in enumerate(bigramssentences)
                            if bigram in bsentence]) >= concepts[i]

        # select only one sentence per cluster
        for clusternum, clustersentences in clusternums.items():
            problem += sum([sentences[j] for j in clustersentences]) <= 1

        # solve the problem
        problem.solve(GLPK(msg=0))

        summary = []
        # get the sentences back
        for j in range(numsentences):
            if sentences[j].varValue == 1:
                summary.append(fullsentences[j])

        return summary

    def summarize(self, wordlimit=100, fusion=True):
        """
        Summarize the documents into one summary
        """
        logger.info("Clustering sentences")
        self.clusterSentences()
        if fusion:
            self.candidates = [_fuseCluster(cluster)
                               for cluster in self.clusters]
        else:
            self.candidates = self.clusters
        self.summary = self._selectSentences(wordlimit)
