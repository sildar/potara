#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import sys
import pulp
import codecs
from collections import Counter
import time
from string import punctuation
from pulp import GLPK
from pulp import GUROBI
import nltk
from nltk.stem import SnowballStemmer
import re
from nltk.corpus import stopwords

def usage(num):
    print("Wrong number of parameters")


def clean(sentence):
    # paste punctuation
    sentence = re.sub(r' ([!.?:,;"\'])', r'\1', sentence)
    sentence = re.sub(r' +',r' ', sentence)
    return sentence


def getClusters(filein):
    """
    Returns a list of clusters from a file.
    """
    clusters = []
    cluster = []
    with codecs.open(filein, 'r', "utf-8") as fin:
        for line in fin.readlines():
            if line.strip() == '':
                clusters.append(cluster)
                cluster = []
            else:
                cluster.append(line)
    # add last cluster
    if cluster != []:
        clusters.append(cluster)
    return clusters
    
TORAW = None
CLUSTERNUM = {}
CLUSTERS = []

def clusternum(sentence):
    """
    Gives the number of the cluster for a given sentence.
    Returns 0 if the sentence belongs to no cluster
    >>> clusternum([["sentence 1 cluster1", "s2 c1"], ["s1 c2"]], "s2 c1")
    1
    """
    global CLUSTERNUM
    global CLUSTERS

    if sentence in CLUSTERNUM:
        return CLUSTERNUM[sentence]

    sentence = wellformatize(sentence)
        
    for i, cluster in enumerate(CLUSTERS):
        for s in cluster:
            if wellformatize(s) == sentence:
                return i
    print("can't find sentence", sentence)
    return -1


def getBigrams(filename, bigramdir, minfreq=3):
    """
    Returns a dictionnary of the bigrams in a list of sentences.
    The dictionary is {bigram : count} where bigram is a pair and count an integer.
    >>> dict(getBigrams(["titi tata"]))
    {('titi', 'tata'): 1}
    """
    bigrams = Counter()
    filename = filename.split('/')[-1]
    if filename.startswith("D"):
        a, _ , _ , d, _ = filename.split('.')
        filename = a.lower() + d.lower()
    with codecs.open(bigramdir + filename, 'r', "utf-8") as fin:
        for line in fin.readlines():
            w1, w2, count = line.split('\t')
            if int(count) >= minfreq:
                bigrams[w1 + ' ' + w2] += int(count[:-1])
    return bigrams

MAXBIGRAMCOUNT = None
BIGRAMCOUNT = None
STEMMER = SnowballStemmer('english')


def weight(concept):
    """
    Gives the weight of a concept given the cluster list.
    """
    global BIGRAMCOUNT
    global MAXBIGRAMCOUNT
    global STEMMER

    # concept = (STEMMER.stem(concept[0]), STEMMER.stem(concept[1]))
    if concept not in BIGRAMCOUNT:
        return 0

    return BIGRAMCOUNT[concept]

def Occ(sentence, concept):
    """
    Returns 1 if the concept belongs to then sentence, else 0
    
    >>> Occ("toto tata titi tutu", ("toto", "tata"))
    1
    
    """
    bigrams = [(w1, w2) for w1,w2 in zip(sentence.split(), sentence.split()[1:])]
    if concept in bigrams:
        return 1
    else:
        return 0

def lenWOpunct(words):
    return sum([1 if word not in punctuation else 0 for word in words])

def hasSentencesFromIdenticalClusters(sentences):
    global CLUSTERS
    global CLUSTERNUM

    nums = [clusternum(s) for s in sentences]
    return len(nums) == len(set(nums))
    

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

def findAlternative(sentence):

    global CLUSTERS

    for cluster in CLUSTERS:
        for s in cluster:
            # find a capitalized alternative
            s = wellformatize(s)
            if s.lower() == sentence and not s.islower():
                print("found alternative")
                return s
    # return the original sentence
    return sentence

    
def main(filein, fileout):


    min_sent_len = 5
    min_doc_freq = 4
    summary_size = 100


    ################################################################################
    # Filtrage des bigrammes par frequence
    ################################################################################
    bigramsfile = "./bigrams_stem_04/"
    doc_freq = getBigrams(filein, bigramsfile, minfreq=min_doc_freq)
    
    bigrams = [bigram.split()[0] + ' ' + bigram.split()[1] for bigram in doc_freq]

    global CLUSTERS
    CLUSTERS = getClusters(filein)
    sentences = [sentence for cluster in CLUSTERS for sentence in cluster if len(sentence.split()) >= min_sent_len]
    delsent = len(sentences)
    # we don't want citations
    sentences = [sentence for sentence in sentences if not ('``' in sentence and 'said' in sentence)]
    delsent -= len(sentences)
    print(delsent)
    
    well_formed_sentences = [wellformatize(sentence) for sentence in sentences]

    # extract bigrams and lowercase
    bigrams_per_sentence = [[t1.lower() + ' ' + t2.lower() for t1,t2 in zip(sentence.split(), sentence.split()[1:])] for sentence in sentences]

    # remove non alphanum bigrams
    bigrams_per_sentence = [[bigram for bigram in s if re.search('[a-zA-Z0-9]', bigram.split()[0]) and re.search('[a-zA-Z0-9]',bigram.split()[1])] for s in bigrams_per_sentence]

    # remove stopwords bigrams
    bigrams_per_sentence = [[bigram for bigram in s if not (bigram.split()[0] in stopwords.words('english') and bigram.split()[1] in stopwords.words('english'))] for s in bigrams_per_sentence]
    
    stemmer = SnowballStemmer('english')
    # stem bigrams
    bigrams_per_sentence = [[stemmer.stem(bigram.split()[0]) + ' ' + stemmer.stem(bigram.split()[1]) for bigram in s] for s in bigrams_per_sentence]


    global CLUSTERNUM
    CLUSTERNUM = {k:v for k,v in zip(range(len(sentences)), map(clusternum, sentences))}
    clusterssent = {}
    for sent,clust in CLUSTERNUM.items():
        if clust in clusterssent:
            clusterssent[clust].append(sent)
        else:
            clusterssent[clust] = [sent]

    ################################################################################
    # Composition du programme ILP
    ################################################################################
    bigrams = doc_freq.keys()
    num_bigrams = len(bigrams)
    num_sentences = len(sentences)


    prob = pulp.LpProblem("Summarization ILP for "+filein , pulp.LpMaximize)

    # Définition des variables
    c = pulp.LpVariable.dicts( name = 'c', 
                          indexs = range(num_bigrams), 
                          lowBound = 0, 
                          upBound = 1,
                          cat = 'Integer' )

    s = pulp.LpVariable.dicts( name = 's', 
                          indexs = range(num_sentences), 
                          lowBound = 0, 
                          upBound = 1,
                          cat = 'Integer' )

    # Ajout de la fonction objective
    prob += sum([ doc_freq[bigrams[i]]*c[i] for i in range(num_bigrams) ])

    # Ajout des contraintes

    ## Contrainte sur la taille du résumé
    prob += sum([s[j] * len(well_formed_sentences[j].split(' ')) for j in range(num_sentences)]) <= summary_size


    ## Contraintes d'intégrité du modèle
    for j in range(num_sentences):
        for i in range(num_bigrams):
            if bigrams[i] in bigrams_per_sentence[j]:
                prob += s[j] <= c[i]

    for i in range(num_bigrams):
        prob += sum( [s[j] for j in range(num_sentences) if bigrams[i] in bigrams_per_sentence[j]] ) >= c[i]

    # Choose capitalized sentence in priority
    for j in range(num_sentences):
        for i in range(num_sentences):
            # compare by removing order on the bigrams
            if set(bigrams_per_sentence[i]) == set(bigrams_per_sentence[j]) and len(well_formed_sentences[j].split(' ')) == len(well_formed_sentences[i].split(' ')):
                if well_formed_sentences[i].islower() and not well_formed_sentences[j].islower():
                    prob += s[i] == 0
                elif well_formed_sentences[j].islower() and not well_formed_sentences[i].islower():
                    prob += s[j] == 0


    # select only one sentence per cluster
    for clus in clusterssent:
        prob += sum([s[j] for j in clusterssent[clus]]) <= 1

        
    # Résolution du problème
    prob.solve(GUROBI(msg = 0))


    summarysentences = []
    # Récupère les phrases
    for j in range(num_sentences):
        if s[j].varValue == 1:
            summarysentences.append(well_formed_sentences[j])

    print(hasSentencesFromIdenticalClusters(summarysentences))
            
    print("Solved with total weight : ", pulp.value(prob.objective))

    summarysentences = map(findAlternative, summarysentences)
    
    with codecs.open(fileout, 'w', 'utf-8') as fout:
        fout.write('\n'.join(summarysentences) + '\n')



if __name__ == "__main__":
    if len(sys.argv) != 3:
        usage(1)
        quit()
    else:
        main(sys.argv[1], sys.argv[2])
