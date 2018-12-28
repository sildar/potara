#!/usr/bin/env python
# -*- coding: utf-8-*-

# go to the root for access to the module
import sys
import os.path
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../')))

import unittest
from potara import document
from potara import similaritymeasures as sm
from potara import summarizer
import gensim


class SummarizerTest(unittest.TestCase):

    def test_init(self):
        s = summarizer.Summarizer()
        testdir = os.path.dirname(os.path.realpath(__file__))
        doc = document.Document(os.path.join(testdir, 'testdata/smalldoc.txt'))
        s.addDocument(doc)
        self.assertEqual(len(s.documents), 1)

    def test_doclogic(self):
        s = summarizer.Summarizer()
        testdir = os.path.dirname(os.path.realpath(__file__))
        doc = document.Document(os.path.join(testdir, 'testdata/smalldoc.txt'))
        s.addDocument(doc)
        s.addDocument(doc)
        self.assertEqual(len(s.documents), 2)
        s.clearDocuments()
        self.assertEqual(s.documents, None)
        s.setDocuments([doc])
        self.assertEqual(len(s.documents), 1)

    def test_clustering(self):
        s = summarizer.Summarizer()
        testdir = os.path.dirname(os.path.realpath(__file__))
        doc = document.Document(os.path.join(testdir, 'testdata/smalldoc.txt'))
        s.addDocument(doc)
        s.addDocument(doc)
        s.clusterSentences()
        self.assertEqual(len(s.clusters), 2)

    def test_fusion(self):
        s1 = "This/DT fake/JJ sentence/NN will/MD " + \
             "create/VB fusions/NNS ./PUNCT"
        s2 = "This/DT awesome/JJ sentence/NN may/MD " + \
             "create/VB fusions/NNS ./PUNCT"

        fusions = summarizer._fuseCluster([s1, s2])
        self.assertTrue(len(fusions) <= 10)
        # if fusion went OK
        self.assertTrue(("this/DT awesome/JJ sentence/NN "
                         "will/MD create/VB fusions/NNS ./PUNCT") in fusions)

    def test_fakefusion(self):
        s1 = "This/DT fake/JJ sentence/NN will/MD " + \
             "create/VB fusions/NNS ./PUNCT"

        fusions = summarizer._fuseCluster([s1])
        self.assertTrue(len(fusions),1)

    def test_selectSentences(self):
        s = summarizer.Summarizer()
        testdir = os.path.dirname(os.path.realpath(__file__))
        doc1 = document.Document(os.path.join(testdir, 'testdata/smalldoc.txt'))
        doc2 = document.Document(os.path.join(testdir, 'testdata/smalldocb.txt'))
        s.addDocument(doc1)
        s.addDocument(doc2)
        maxwords = 50
        s.summarize(maxwords)
        print(s.summary)


class DocumentTest(unittest.TestCase):

    def test_normalize(self):

        t1 = """ This is   a .strange
        sentence that has \t\tlots\n
        of spaces\t\n """
        et1 = "This is a .strange sentence that has lots of spaces"
        pt1 = document.normalize(t1)

        self.assertEqual(pt1, et1)

    def test_stem(self):
        words = ["these", "sentences", "are", "awesome"]
        estems = ["these", "sentenc", "are", "awesom"]

        pstems = [document.stem(word) for word in words]
        self.assertEqual(estems, pstems)

    def test_sentTokenize(self):

        t1 = "This test is composed of several sentences. Some of " + \
             "them contain abbreviations like POS. POS means Part Of " + \
             "Speech by the way."

        esent1 = ["This test is composed of several sentences.",
                  "Some of them contain abbreviations like POS.",
                  "POS means Part Of Speech by the way."]
        psent1 = document.sentTokenize(t1)
        self.assertEqual(esent1, psent1)

    def test_postag(self):

        s1 = "I wonder how this sentence will be postagged .".split()

        es1 = [(u'I', u'PRP'), (u'wonder', u'VBP'), (u'how', u'WRB'),
               (u'this', u'DT'), (u'sentence', u'NN'), (u'will', u'MD'),
               (u'be', u'VB'), (u'postagged', u'VBN'), (u'.', 'PUNCT')]
        ps1 = document.postag(s1)
        self.assertEqual(es1, ps1)

    def test_docinit(self):
        testdir = os.path.dirname(os.path.realpath(__file__))
        docfile1 = testdir + "/testdata/smalldoc.txt"
        doc1 = document.Document(docfile1)

        edoc1stem = [[(u'clean', u'JJ'), (u'document', u'NN')],
                     [(u'short', u'JJ'), (u'illform', u'JJ')]]
        pdoc1stem = doc1.stemTokens
        self.assertEqual(edoc1stem, pdoc1stem)


class SimilarityTest(unittest.TestCase):

    def test_cosine(self):
        s1 = "This sentence is not right."
        s2 = "This sentence is wrong."

        ecos = 0.671
        pcos = sm.cosine(s1, s2)

        self.assertAlmostEqual(ecos, pcos, places=3)

    def test_cosine_empty(self):
        s1 = "This sentence is OK ."
        s2 = ""

        ecos = 0
        pcos = sm.cosine(s1, s2)
        self.assertEqual(ecos, pcos)

    # w2v needs a model.
    # Without it, I'll just pass the tests (useful for travis)
    def test_w2v(self):
        testdir = os.path.dirname(os.path.realpath(__file__))
        modelfile = testdir + '/../potara/data/enwiki9stempos.model'
        try:
            model = gensim.models.word2vec.Word2Vec.load(modelfile)
            esim = 0.9
        except Exception as e:
            # mock a similarity model
            class FakeModel():
                class FakeWv():
                    vocab = []
                    sim = {}

                    def __contains__(self, item):
                        return item in self.vocab

                    def similarity(self, w1, w2):
                        if w1 + '_' + w2 in self.sim:
                            return self.sim[w1 + '_' + w2]
                        else:
                            return self.sim[w2 + '_' + w1]

                wv = FakeWv()

                def __init__(self):
                    pass

            model = FakeModel()
            model.wv.vocab = ['right/JJ', 'wrong/JJ']
            model.wv.sim = {'right/JJ_wrong/JJ': 0.5}
            esim = 0.9166

        s1 = "This/T beautiful/JJ sentence/NN is/V not/N right/JJ ./PUNCT"
        s2 = "This/T beautiful/JJ sentence/NN is/V wrong/JJ ./PUNCT"

        psim = sm.w2v(s1, s2, model)
        self.assertAlmostEqual(esim, psim, places=1)

        # order doesn't matter
        psim2 = sm.w2v(s2, s1, model)
        self.assertEqual(psim, psim2)

    def test_w2v_notinvocab(self):
        testdir = os.path.dirname(os.path.realpath(__file__))
        modelfile = testdir + '/../potara/data/enwiki9stempos.model'
        try:
            model = gensim.models.word2vec.Word2Vec.load(modelfile)
        except:
            return

        s1 = "This/T beauful/JJ sentence/NN is/V not/N right/JJ ./PUNCT"
        s2 = "This/T beautiful/JJ sentence/NN is/V wrong/JJ ./PUNCT"

        esim = 0.8
        psim = sm.w2v(s1, s2, model)
        self.assertAlmostEqual(esim, psim, places=1)

    def test_w2v_singleword(self):
        testdir = os.path.dirname(os.path.realpath(__file__))
        modelfile = testdir + '/../potara/data/enwiki9stempos.model'
        try:
            model = gensim.models.word2vec.Word2Vec.load(modelfile)
        except:
            return

        s1 = "right/JJ"
        s2 = "wrong/JJ"

        # a single different word means 0 sim
        esim = 0
        psim = sm.w2v(s1, s2, model)
        self.assertEqual(esim, psim)

        s3 = "right/JJ"
        esim2 = 1
        psim2 = sm.w2v(s1, s3, model)
        self.assertEqual(esim2, psim2)

    def test_w2v_untag(self):
        testdir = os.path.dirname(os.path.realpath(__file__))
        modelfile = testdir + '/../potara/data/enwiki9stempos.model'
        try:
            model = gensim.models.word2vec.Word2Vec.load(modelfile)
        except:
            return

        s1 = "This sentence is not right ."
        s2 = "This sentence is wrong ."

        # without tags we consider the intersection over min length
        esim = 4.0/5
        psim = sm.w2v(s1, s2, model)
        self.assertEqual(esim, psim)


if __name__ == '__main__':
    unittest.main()
        