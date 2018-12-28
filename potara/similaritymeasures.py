#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Contains several similarity measures.
These measures deal with raw strings which words
are supposed to be white-space separated.
"""

from collections import Counter
from operator import itemgetter
import math


def cosine(s1, s2):
    """
    Retuns the cosine value between two strings

    >>> cosine("This is a sentence", "This is a sentence")
    1.0
    """

    vec1 = Counter(s1.split())
    vec2 = Counter(s2.split())

    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def w2v(s1, s2, wordmodel):
    """
    Calculates the similarity between two strings
    given a word model that gives word-word similarity.
    The word model is supposed to hold a vocab field
    that contains the vocabulary.
    It must have a similarity(word1, word2) method.
    """

    if s1 == s2:
        return 1.0

    intersection = set(s1.split()) & set(s2.split())

    # give 1 point per common word
    commonwords = len(intersection)

    # We want at least one common word
    if commonwords == 0:
        return 0

    # remove common words
    l1 = [word for word in s1.split() if word not in intersection]
    l2 = [word for word in s2.split() if word not in intersection]

    # change order depending on size
    if len(l1) > len(l2):
        l1, l2 = l2, l1

    totalscore = 0

    for t1 in l1:
        sublist = []
        hasitem = False
        for i, t2 in enumerate(l2):
            # check if POS are here
            if len(t1.split('/')) > 1:
                # compare same POS words
                if t1.split('/')[1][:2] == t2.split('/')[1][:2]:
                    if t1 in wordmodel.wv and t2 in wordmodel.wv:
                        sublist.append((i, wordmodel.wv.similarity(t1, t2)))
                        hasitem = True
                    # if we don't know one of the words
                    # consider them as dissimilar
                    else:
                        sublist.append((i, 0))
                else:
                    sublist.append((i, 0))

        if hasitem:
            maxitem, subscore = max(sublist, key=itemgetter(1))
            l2.pop(maxitem)
            totalscore += subscore

    num = float(commonwords + totalscore)
    denum = min(len(s1.split()), len(s2.split()))
    score = num / denum

    return score
