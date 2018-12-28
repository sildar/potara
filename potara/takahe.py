#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
:Name:
    takahe

:Authors:
    Florian Boudin (florian.boudin@univ-nantes.fr)

:Version:
    0.4

:Date:
    Mar. 2013

:Description:
    takahe is a multi-sentence compression module. Given a set of redundant 
    sentences, a word-graph is constructed by iteratively adding sentences to 
    it. The best compression is obtained by finding the shortest path in the
    word graph. The original algorithm was published and described in
    [filippova:2010:COLING]_. A keyphrase-based reranking method, described in
    [boudin-morin:2013:NAACL]_ can be applied to generate more informative 
    compressions.

    .. [filippova:2010:COLING] Katja Filippova, Multi-Sentence Compression: 
       Finding Shortest Paths in Word Graphs, *Proceedings of the 23rd 
       International Conference on Computational Linguistics (Coling 2010)*, 
       pages 322-330, 2010.
    .. [boudin-morin:2013:NAACL] Florian Boudin and Emmanuel Morin, Keyphrase 
       Extraction for N-best Reranking in Multi-Sentence Compression, 
       *Proceedings of the 2013 Conference of the North American Chapter of the
       Association for Computational Linguistics: Human Language Technologies 
       (NAACL-HLT 2013)*, 2013.


:History:
    Development history of the takahe module:
        - 0.4 (Mar. 2013) adding the keyphrase-based nbest reranking algorithm
        - 0.33 (Feb. 2013), bug fixes and better code documentation
        - 0.32 (Jun. 2012), Punctuation marks are now considered within the 
          graph, compressions are then punctuated
        - 0.31 (Nov. 2011), modified context function (uses the left and right 
          contexts), improved docstring documentation, bug fixes
        - 0.3 (Oct. 2011), improved K-shortest paths algorithm including 
          verb/size constraints and ordered lists for performance
        - 0.2 (Dec. 2010), removed dependencies from nltk (i.e. POS-tagging, 
          tokenization and stopwords removal)
        - 0.1 (Nov. 2010), first version

:Dependencies:
    The following Python modules are required:
        - `networkx <http://networkx.github.com/>`_ for the graph construction
          (v1.2+)

:Usage:
    A typical usage of this module is::
    
        import takahe
        
        # A list of tokenized and POS-tagged sentences
        sentences = ['Hillary/NNP Clinton/NNP wanted/VBD to/stop visit/VB ...']
        
        # Create a word graph from the set of sentences with parameters :
        # - minimal number of words in the compression : 6
        # - language of the input sentences : en (english)
        # - POS tag for punctuation marks : PUNCT
        compresser = takahe.word_graph( sentences, 
                                        nb_words = 6, 
                                        lang = 'en', 
                                        punct_tag = "PUNCT" )

        # Get the 50 best paths
        candidates = compresser.get_compression(50)

        # 1. Rerank compressions by path length (Filippova's method)
        for cummulative_score, path in candidates:

            # Normalize path score by path length
            normalized_score = cummulative_score / len(path)

            # Print normalized score and compression
            print round(normalized_score, 3), ' '.join([u[0] for u in path])

        # Write the word graph in the dot format
        compresser.write_dot('test.dot')

        # 2. Rerank compressions by keyphrases (Boudin and Morin's method)
        reranker = takahe.keyphrase_reranker( sentences,  
                                              candidates, 
                                              lang = 'en' )

        reranked_candidates = reranker.rerank_nbest_compressions()

        # Loop over the best reranked candidates
        for score, path in reranked_candidates:
            
            # Print the best reranked candidates
            print round(score, 3), ' '.join([u[0] for u in path])

:Misc:
    The Takahe is a flightless bird indigenous to New Zealand. It was thought to
    be extinct after the last four known specimens were taken in 1898. However, 
    after a carefully planned search effort the bird was rediscovered by on 
    November 20, 1948. (Wikipedia, http://en.wikipedia.org/wiki/takahe)  
"""

import math
import codecs
import os
import re
import sys
import bisect
import networkx as nx
#import matplotlib.pyplot as plt

#~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
# [ Class word_graph
#~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
class word_graph:
    """
    The word_graph class constructs a word graph from the set of sentences given
    as input. The set of sentences is a list of strings, sentences are tokenized
    and words are POS-tagged (e.g. ``"Saturn/NNP is/VBZ the/DT sixth/JJ 
    planet/NN from/IN the/DT Sun/NNP in/IN the/DT Solar/NNP System/NNP"``). 
    Four optional parameters can be specified:

    - nb_words is is the minimal number of words for the best compression 
      (default value is 8).
    - lang is the language parameter and is used for selecting the correct 
      stopwords list (default is "en" for english, stopword lists are localized 
      in /resources/ directory).
    - punct_tag is the punctuation mark tag used during graph construction 
      (default is PUNCT).
    """

    #-T-----------------------------------------------------------------------T-
    def __init__(self, sentence_list, nb_words=8, lang="en", punct_tag="PUNCT", pos_sep="/"):

        self.sentence = list(sentence_list)
        """ A list of sentences provided by the user. """

        self.length = len(sentence_list)
        """ The number of sentences given for fusion. """
        
        self.nb_words = nb_words
        """ The minimal number of words in the compression. """

        self.resources = os.path.dirname(__file__) + '/resources/'
        """ The path of the resources folder. """

        self.stopword_path = self.resources+'stopwords.'+lang+'.dat'
        """ The path of the stopword list, e.g. stopwords.[lang].dat. """

        self.stopwords = self.load_stopwords(self.stopword_path)
        """ The set of stopwords loaded from stopwords.[lang].dat. """

        self.punct_tag = punct_tag
        """ The stopword tag used in the graph. """

        self.graph = nx.DiGraph()
        """ The directed graph used for fusion. """
    
        self.start = '-start-'
        """ The start token in the graph. """

        self.stop = '-end-'
        """ The end token in the graph. """

        self.sep = '/-/'
        """ The separator used between a word and its POS in the graph. """

        self.pos_sep = pos_sep
        
        self.term_freq = {}
        """ The frequency of a given term. """
        
        self.verbs = set(['VB', 'VBD', 'VBP', 'VBZ', 'VH', 'VHD', 'VHP', 'VBZ', 
        'VV', 'VVD', 'VVP', 'VVZ'])
        """
        The list of verb POS tags required in the compression. At least *one* 
        verb must occur in the candidate compressions.
        """

        # Replacing default values for French
        if lang == "fr":
            self.verbs = set(['V', 'VPP', 'VINF'])

        # 1. Pre-process the sentences
        self.pre_process_sentences()

        # 2. Compute term statistics
        self.compute_statistics()

        # 3. Build the word graph
        self.build_graph()
    #-B-----------------------------------------------------------------------B-


    #-T-----------------------------------------------------------------------T-
    def pre_process_sentences(self):
        """
        Pre-process the list of sentences given as input. Split sentences using 
        whitespaces and convert each sentence to a list of (word, POS) tuples.
        """

        for i in range(self.length):
        
            # Normalise extra white spaces
            self.sentence[i] = re.sub(' +', ' ', self.sentence[i])
            self.sentence[i] = self.sentence[i].strip()
            
            # Tokenize the current sentence in word/POS
            sentence = self.sentence[i].split(' ')

            # Creating an empty container for the cleaned up sentence
            container = [(self.start, self.start)]

            # Looping over the words
            for w in sentence:
                
                # Splitting word, POS
                m = re.match("^(.+)/(.+)$", w)
                
                # Extract the word information
                token, POS = m.group(1), m.group(2)

                # Add the token/POS to the sentence container
                container.append((token.lower(), POS))
                    
            # Add the stop token at the end of the container
            container.append((self.stop, self.stop))

            # Recopy the container into the current sentence
            self.sentence[i] = container
    #-B-----------------------------------------------------------------------B-
    
    
    #-T-----------------------------------------------------------------------T-
    def build_graph(self):
        """
        Constructs a directed word graph from the list of input sentences. Each
        sentence is iteratively added to the directed graph according to the 
        following algorithm:

        - Word mapping/creation is done in four steps:

            1. non-stopwords for which no candidate exists in the graph or for 
               which an unambiguous mapping is possible or which occur more than
               once in the sentence

            2. non-stopwords for which there are either several possible
               candidates in the graph

            3. stopwords

            4. punctuation marks

        For the last three groups of words where mapping is ambiguous we check 
        the immediate context (the preceding and following words in the sentence 
        and the neighboring nodes in the graph) and select the candidate which 
        has larger overlap in the context, or the one with a greater frequency 
        (i.e. the one which has more words mapped onto it). Stopwords are mapped 
        only if there is some overlap in non-stopwords neighbors, otherwise a 
        new node is created. Punctuation marks are mapped only if the preceding 
        and following words in the sentence and the neighboring nodes are the
        same.

        - Edges are then computed and added between mapped words.
        
        Each node in the graph is represented as a tuple ('word/POS', id) and 
        possesses an info list containing (sentence_id, position_in_sentence)
        tuples.
        """     

        # Iteratively add each sentence in the graph ---------------------------
        for i in range(self.length):

            # Compute the sentence length
            sentence_len = len(self.sentence[i])

            # Create the mapping container
            mapping = [0] * sentence_len

            #-------------------------------------------------------------------
            # 1. non-stopwords for which no candidate exists in the graph or for 
            #    which an unambiguous mapping is possible or which occur more 
            #    than once in the sentence.
            #-------------------------------------------------------------------
            for j in range(sentence_len):

                # Get the word and tag
                token, POS = self.sentence[i][j]

                # If stopword or punctuation mark, continues
                if token in self.stopwords or re.search('(?u)^\W$', token):
                    continue
            
                # Create the node identifier
                node = token.lower() + self.sep + POS

                # Find the number of ambiguous nodes in the graph
                k = self.ambiguous_nodes(node)

                # If there is no node in the graph, create one with id = 0
                if k == 0:

                    # Add the node in the graph
                    self.graph.add_node( (node, 0), info=[(i, j)],
                                         label=token.lower() )

                    # Mark the word as mapped to k
                    mapping[j] = (node, 0)

                # If there is only one matching node in the graph (id is 0)
                elif k == 1:

                    # Get the sentences id of this node
                    ids = []
                    for sid, pos_s in self.graph.node[(node, 0)]['info']:
                        ids.append(sid)
                    
                    # Update the node in the graph if not same sentence
                    if not i in ids:
                        self.graph.node[(node, 0)]['info'].append((i, j))
                        mapping[j] = (node, 0)

                    # Else Create new node for redundant word
                    else:
                        self.graph.add_node( (node, 1), info=[(i, j)], 
                                             label=token.lower() )
                        mapping[j] = (node, 1)

            #-------------------------------------------------------------------
            # 2. non-stopwords for which there are either several possible
            #    candidates in the graph.
            #-------------------------------------------------------------------
            for j in range(sentence_len):

                # Get the word and tag
                token, POS = self.sentence[i][j]
                
                # If stopword or punctuation mark, continues
                if token in self.stopwords or re.search('(?u)^\W$', token):
                    continue

                # If word is not already mapped to a node
                if mapping[j] == 0:

                    # Create the node identifier
                    node = token.lower() + self.sep + POS
                    
                    # Create the neighboring nodes identifiers
                    prev_token, prev_POS = self.sentence[i][j-1]
                    next_token, next_POS = self.sentence[i][j+1]
                    prev_node = prev_token.lower() + self.sep + prev_POS
                    next_node = next_token.lower() + self.sep + next_POS
                    
                    # Find the number of ambiguous nodes in the graph
                    k = self.ambiguous_nodes(node)

                    # Search for the ambiguous node with the larger overlap in
                    # context or the greater frequency.
                    ambinode_overlap = []
                    ambinode_frequency = []
            
                    # For each ambiguous node
                    for l in range(k):

                        # Get the immediate context words of the nodes
                        l_context = self.get_directed_context(node, l, 'left')
                        r_context = self.get_directed_context(node, l, 'right')
                        
                        # Compute the (directed) context sum
                        val = l_context.count(prev_node) 
                        val += r_context.count(next_node)

                        # Add the count of the overlapping words
                        ambinode_overlap.append(val)

                        # Add the frequency of the ambiguous node
                        ambinode_frequency.append(
                            len( self.graph.node[(node, l)]['info'] )
                        )
                
                    # Search for the best candidate while avoiding a loop
                    found = False
                    selected = 0
                    while not found:
                    
                        # Select the ambiguous node
                        selected = self.max_index(ambinode_overlap)
                        if ambinode_overlap[selected] == 0:
                            selected = self.max_index(ambinode_frequency)
                        
                        # Get the sentences id of this node
                        ids = []
                        for sid, p in self.graph.node[(node, selected)]['info']:
                            ids.append(sid)
                        
                        # Test if there is no loop
                        if i not in ids:
                            found = True
                            break
            
                        # Remove the candidate from the lists
                        else:
                            del ambinode_overlap[selected]
                            del ambinode_frequency[selected]
                            
                        # Avoid endless loops
                        if len(ambinode_overlap) == 0:
                            break
                    
                    # Update the node in the graph if not same sentence
                    if found:
                        self.graph.node[(node, selected)]['info'].append((i, j))
                        mapping[j] = (node, selected)

                    # Else create new node for redundant word
                    else:
                        self.graph.add_node( (node, k), info=[(i, j)], 
                                             label=token.lower() )
                        mapping[j] = (node, k)
            
            #-------------------------------------------------------------------
            # 3. map the stopwords to the nodes
            #-------------------------------------------------------------------
            for j in range(sentence_len):

                # Get the word and tag
                token, POS = self.sentence[i][j]

                # If *NOT* stopword, continues
                if not token in self.stopwords :
                    continue

                # Create the node identifier
                node = token.lower() + self.sep + POS
                    
                # Find the number of ambiguous nodes in the graph
                k = self.ambiguous_nodes(node)

                # If there is no node in the graph, create one with id = 0
                if k == 0:

                    # Add the node in the graph
                    self.graph.add_node( (node, 0), info=[(i, j)], 
                                         label=token.lower() )

                    # Mark the word as mapped to k
                    mapping[j] = (node, 0)
   
                # Else find the node with overlap in context or create one
                else:
                    
                    # Create the neighboring nodes identifiers
                    prev_token, prev_POS = self.sentence[i][j-1]
                    next_token, next_POS = self.sentence[i][j+1]
                    prev_node = prev_token.lower() + self.sep + prev_POS
                    next_node = next_token.lower() + self.sep + next_POS

                    ambinode_overlap = []
            
                    # For each ambiguous node
                    for l in range(k):

                        # Get the immediate context words of the nodes, the
                        # boolean indicates to consider only non stopwords
                        l_context = self.get_directed_context(node, l, 'left',\
                                    True)
                        r_context = self.get_directed_context(node, l, 'right',\
                                    True)
                        
                        # Compute the (directed) context sum
                        val = l_context.count(prev_node) 
                        val += r_context.count(next_node)

                        # Add the count of the overlapping words
                        ambinode_overlap.append(val)
                    
                    # Get best overlap candidate
                    selected = self.max_index(ambinode_overlap)
                
                    # Get the sentences id of the best candidate node
                    ids = []
                    for sid, pos_s in self.graph.node[(node, selected)]['info']:
                        ids.append(sid)

                    # Update the node in the graph if not same sentence and 
                    # there is at least one overlap in context
                    if i not in ids and ambinode_overlap[selected] > 0:
                    # if i not in ids and \
                    # (ambinode_overlap[selected] > 1 and POS==self.punct_tag) or\
                    # (ambinode_overlap[selected] > 0 and POS!=self.punct_tag) :

                        # Update the node in the graph
                        self.graph.node[(node, selected)]['info'].append((i, j))

                        # Mark the word as mapped to k
                        mapping[j] = (node, selected)

                    # Else create a new node
                    else:
                        # Add the node in the graph
                        self.graph.add_node( (node, k) , info=[(i, j)],
                                             label=token.lower() )

                        # Mark the word as mapped to k
                        mapping[j] = (node, k)

            #-------------------------------------------------------------------
            # 4. lasty map the punctuation marks to the nodes
            #-------------------------------------------------------------------
            for j in range(sentence_len):

                # Get the word and tag
                token, POS = self.sentence[i][j]

                # If *NOT* punctuation mark, continues
                if not re.search('(?u)^\W$', token):
                    continue

                # Create the node identifier
                node = token.lower() + self.sep + POS
                    
                # Find the number of ambiguous nodes in the graph
                k = self.ambiguous_nodes(node)

                # If there is no node in the graph, create one with id = 0
                if k == 0:

                    # Add the node in the graph
                    self.graph.add_node( (node, 0), info=[(i, j)], 
                                         label=token.lower() )

                    # Mark the word as mapped to k
                    mapping[j] = (node, 0)
   
                # Else find the node with overlap in context or create one
                else:
                    
                    # Create the neighboring nodes identifiers
                    prev_token, prev_POS = self.sentence[i][j-1]
                    next_token, next_POS = self.sentence[i][j+1]
                    prev_node = prev_token.lower() + self.sep + prev_POS
                    next_node = next_token.lower() + self.sep + next_POS

                    ambinode_overlap = []
            
                    # For each ambiguous node
                    for l in range(k):

                        # Get the immediate context words of the nodes
                        l_context = self.get_directed_context(node, l, 'left')
                        r_context = self.get_directed_context(node, l, 'right')
                        
                        # Compute the (directed) context sum
                        val = l_context.count(prev_node) 
                        val += r_context.count(next_node)

                        # Add the count of the overlapping words
                        ambinode_overlap.append(val)
                    
                    # Get best overlap candidate
                    selected = self.max_index(ambinode_overlap)
                
                    # Get the sentences id of the best candidate node
                    ids = []
                    for sid, pos_s in self.graph.node[(node, selected)]['info']:
                        ids.append(sid)

                    # Update the node in the graph if not same sentence and 
                    # there is at least one overlap in context
                    if i not in ids and ambinode_overlap[selected] > 1:

                        # Update the node in the graph
                        self.graph.node[(node, selected)]['info'].append((i, j))

                        # Mark the word as mapped to k
                        mapping[j] = (node, selected)

                    # Else create a new node
                    else:
                        # Add the node in the graph
                        self.graph.add_node( (node, k), info=[(i, j)], 
                                             label=token.lower() )

                        # Mark the word as mapped to k
                        mapping[j] = (node, k)

            #-------------------------------------------------------------------
            # 4. Connects the mapped words with directed edges
            #-------------------------------------------------------------------
            for j in range(1, len(mapping)):
                self.graph.add_edge(mapping[j-1], mapping[j])

        # Assigns a weight to each node in the graph ---------------------------
        for node1, node2 in self.graph.edges_iter():
            edge_weight = self.get_edge_weight(node1, node2)
            self.graph.add_edge(node1, node2, weight=edge_weight)
    #-B-----------------------------------------------------------------------B-

 
    #-T-----------------------------------------------------------------------T-
    def ambiguous_nodes(self, node):
        """
        Takes a node in parameter and returns the number of possible candidate 
        (ambiguous) nodes in the graph.
        """
        k = 0
        while(self.graph.has_node((node, k))):
            k += 1
        return k
    #-B-----------------------------------------------------------------------B-


    #-T-----------------------------------------------------------------------T-
    def get_directed_context(self, node, k, dir='all', non_pos=False):
        """
        Returns the directed context of a given node, i.e. a list of word/POS of
        the left or right neighboring nodes in the graph. The function takes 
        four parameters :

        - node is the word/POS tuple
        - k is the node identifier used when multiple nodes refer to the same 
          word/POS (e.g. k=0 for (the/DET, 0), k=1 for (the/DET, 1), etc.)
        - dir is the parameter that controls the directed context calculation, 
          it can be set to left, right or all (default)
        - non_pos is a boolean allowing to remove stopwords from the context 
          (default is false)
        """

        # Define the context containers
        l_context = []
        r_context = []

        # For all the sentence/position tuples
        for sid, off in self.graph.node[(node, k)]['info']:
            
            prev = self.sentence[sid][off-1][0].lower() + self.sep +\
                   self.sentence[sid][off-1][1]
                   
            next = self.sentence[sid][off+1][0].lower() + self.sep +\
                   self.sentence[sid][off+1][1]
                   
            if non_pos:
                if self.sentence[sid][off-1][0] not in self.stopwords:
                    l_context.append(prev)
                if self.sentence[sid][off+1][0] not in self.stopwords:
                    r_context.append(next)
            else:
                l_context.append(prev)
                r_context.append(next)

        # Returns the left (previous) context
        if dir == 'left':
            return l_context
        # Returns the right (next) context
        elif dir == 'right':
            return r_context
        # Returns the whole context
        else:
            l_context.extend(r_context)
            return l_context
    #-B-----------------------------------------------------------------------B-


    #-T-----------------------------------------------------------------------T-
    def get_edge_weight(self, node1, node2):
        """
        Compute the weight of an edge *e* between nodes *node1* and *node2*. It 
        is computed as e_ij = (A / B) / C with:
        
        - A = freq(i) + freq(j), 
        - B = Sum (s in S) 1 / diff(s, i, j)
        - C = freq(i) * freq(j)
        
        A node is a tuple of ('word/POS', unique_id).
        """

        # Get the list of (sentence_id, pos_in_sentence) for node1
        info1 = self.graph.node[node1]['info']
        
        # Get the list of (sentence_id, pos_in_sentence) for node2
        info2 = self.graph.node[node2]['info']
        
        # Get the frequency of node1 in the graph
        # freq1 = self.graph.degree(node1)
        freq1 = len(info1)
        
        # Get the frequency of node2 in cluster
        # freq2 = self.graph.degree(node2)
        freq2 = len(info2)

        # Initializing the diff function list container
        diff = []

        # For each sentence of the cluster (for s in S)
        for s in range(self.length):
        
            # Compute diff(s, i, j) which is calculated as
            # pos(s, i) - pos(s, j) if pos(s, i) < pos(s, j)
            # O otherwise
    
            # Get the positions of i and j in s, named pos(s, i) and pos(s, j)
            # As a word can appear at multiple positions in a sentence, a list
            # of positions is used
            pos_i_in_s = []
            pos_j_in_s = []
            
            # For each (sentence_id, pos_in_sentence) of node1
            for sentence_id, pos_in_sentence in info1:
                # If the sentence_id is s
                if sentence_id == s:
                    # Add the position in s
                    pos_i_in_s.append(pos_in_sentence)
            
            # For each (sentence_id, pos_in_sentence) of node2
            for sentence_id, pos_in_sentence in info2:
                # If the sentence_id is s
                if sentence_id == s:
                    # Add the position in s
                    pos_j_in_s.append(pos_in_sentence)
                    
            # Container for all the diff(s, i, j) for i and j
            all_diff_pos_i_j = []
            
            # Loop over all the i, j couples
            for x in range(len(pos_i_in_s)):
                for y in range(len(pos_j_in_s)):
                    diff_i_j = pos_i_in_s[x] - pos_j_in_s[y]
                    # Test if word i appears *BEFORE* word j in s
                    if diff_i_j < 0:
                        all_diff_pos_i_j.append(-1.0*diff_i_j)
                        
            # Add the mininum distance to diff (i.e. in case of multiple 
            # occurrencies of i or/and j in sentence s), 0 otherwise.
            if len(all_diff_pos_i_j) > 0:
                diff.append(1.0/min(all_diff_pos_i_j))
            else:
                diff.append(0.0)
                
        weight1 = freq1
        weight2 = freq2

        return ( (freq1 + freq1) / sum(diff) ) / (weight1 * weight2)
    #-B-----------------------------------------------------------------------B-
   
   
    #-T-----------------------------------------------------------------------T-
    def k_shortest_paths(self, start, end, k=10):
        """
        Simple implementation of a k-shortest paths algorithms. Takes three
        parameters: the starting node, the ending node and the number of 
        shortest paths desired. Returns a list of k tuples (path, weight).
        """

        # Initialize the list of shortest paths
        kshortestpaths = []

        # Initializing the label container 
        orderedX = []
        orderedX.append((0, start, 0))
        
        # Initializing the path container
        paths = {}
        paths[(0, start, 0)] = [start]
        
        # Initialize the visited container
        visited = {}
        visited[start] = 0

        # Initialize the sentence container that will be used to remove 
        # duplicate sentences passing throught different nodes
        sentence_container = {}
    
        # While the number of shortest paths isn't reached or all paths explored
        while len(kshortestpaths) < k and len(orderedX) > 0:

            # Searching for the shortest distance in orderedX
            shortest = orderedX.pop(0)
            shortestpath = paths[shortest]
            
            # Removing the shortest node from X and paths
            del paths[shortest]
    
            # Iterating over the accessible nodes
            for node in self.graph.neighbors(shortest[1]):
            
                # To avoid loops
                if node in shortestpath:
                    continue
            
                # Compute the weight to node
                w = shortest[0] + self.graph[shortest[1]][node]['weight']
            
                # If found the end, adds to k-shortest paths 
                if node == end:

                    #-T-------------------------------------------------------T-
                    # --- Constraints on the shortest paths

                    # 1. Check if path contains at least one werb
                    # 2. Check the length of the shortest path, without 
                    #    considering punctuation marks and starting node (-1 in
                    #    the range loop, because nodes are reversed)
                    # 3. Check the paired parentheses and quotation marks
                    # 4. Check if sentence is not redundant

                    nb_verbs = 0
                    length = 0
                    paired_parentheses = 0
                    quotation_mark_number = 0
                    raw_sentence = ''

                    for i in range(len(shortestpath) - 1):
                        word, tag = shortestpath[i][0].split(self.sep)
                        # 1.
                        if tag in self.verbs:
                            nb_verbs += 1
                        # 2.
                        if not re.search('(?u)^\W$', word):
                            length += 1
                        # 3.
                        else:
                            if word == '(':
                                paired_parentheses -= 1
                            elif word == ')':
                                paired_parentheses += 1
                            elif word == '"':
                                quotation_mark_number += 1
                        # 4.
                        raw_sentence += word + ' '
                    
                    # Remove extra space from sentence
                    raw_sentence = raw_sentence.strip()

                    if nb_verbs >0 and \
                        length >= self.nb_words and \
                        paired_parentheses == 0 and \
                        (quotation_mark_number%2) == 0 \
                        and raw_sentence not in sentence_container:
                        path = [node]
                        path.extend(shortestpath)
                        path.reverse()
                        weight = float(w) #/ float(length)
                        kshortestpaths.append((path, weight))
                        sentence_container[raw_sentence] = 1

                    #-B-------------------------------------------------------B-

                else:
            
                    # test if node has already been visited
                    if node in visited:
                        visited[node] += 1
                    else:
                        visited[node] = 0
                    id = visited[node]

                    # Add the node to orderedX
                    bisect.insort(orderedX, (w, node, id))
                    
                    # Add the node to paths
                    paths[(w, node, id)] = [node]
                    paths[(w, node, id)].extend(shortestpath)
    
        # Returns the list of shortest paths
        return kshortestpaths
    #-B-----------------------------------------------------------------------B-
    
    #-T-----------------------------------------------------------------------T-
    def get_compression(self, nb_candidates=50):
        """
        Searches all possible paths from **start** to **end** in the word graph,
        removes paths containing no verb or shorter than *n* words. Returns an
        ordered list (smaller first) of nb (default value is 50) (cummulative 
        score, path) tuples. The score is not normalized with the sentence 
        length.
        """

        for sentence in self.sentence:
            hasVerb = False
            for word, tag in sentence:
                if tag in self.verbs:
                    hasVerb = True
        if not hasVerb:
            raise Exception("No verb in the provided sentences")

        # Search for the k-shortest paths in the graph
        self.paths = self.k_shortest_paths((self.start+self.sep+self.start, 0),
                                           (self.stop+self.sep+self.stop, 0),
                                            nb_candidates)

        # Initialize the fusion container
        fusions = []
        
        # Test if there are some paths
        if len(self.paths) > 0:
        
            # For nb candidates 
            for i in range(min(nb_candidates, len(self.paths))):
                nodes = self.paths[i][0]
                sentence = []
                
                for j in range(1, len(nodes)-1):
                    word, tag = nodes[j][0].split(self.sep)
                    sentence.append((word, tag))

                bisect.insort(fusions, (self.paths[i][1], sentence))

        return fusions
    #-B-----------------------------------------------------------------------B-

    #-T-----------------------------------------------------------------------T-
    def max_index(self, l):
        """ Returns the index of the maximum value of a given list. """

        ll = len(l)
        if ll < 0:
            return None
        elif ll == 1:
            return 0
        max_val = l[0]
        max_ind = 0
        for z in range(1, ll):
            if l[z] > max_val:
                max_val = l[z]
                max_ind = z
        return max_ind
    #-B-----------------------------------------------------------------------B-


    #-T-----------------------------------------------------------------------T-
    def compute_statistics(self):
        """
        This function iterates over the cluster's sentences and computes the
        following statistics about each word:
        
        - term frequency (self.term_freq)
        """

        # Structure for containing the list of sentences in which a term occurs
        terms = {}

        # Loop over the sentences
        for i in range(self.length):
        
            # For each tuple (token, POS) of sentence i
            for token, POS in self.sentence[i]:
            
                # generate the word/POS token
                node = token.lower() + self.sep + POS
                
                # Add the token to the terms list
                if node not in terms:
                    terms[node] = [i]
                else:
                    terms[node].append(i)

        # Loop over the terms
        for w in terms:

            # Compute the term frequency
            self.term_freq[w] = len(terms[w])
    #-B-----------------------------------------------------------------------B-


    #-T-----------------------------------------------------------------------T-
    def load_stopwords(self, path):
        """
        This function loads a stopword list from the *path* file and returns a 
        set of words. Lines begining by '#' are ignored.
        """

        # Set of stopwords
        stopwords = set([])

        # For each line in the file
        with codecs.open(path, 'r', 'utf-8') as f:
            for line in f:
                if not re.search('^#', line) and len(line.strip()) > 0:
                    stopwords.add(line.strip().lower())

        # Return the set of stopwords
        return stopwords
    #-B-----------------------------------------------------------------------B-
    

    #-T-----------------------------------------------------------------------T-
    def write_dot(self, dotfile):
        """ Outputs the word graph in dot format in the specified file. """
        nx.write_dot(self.graph, dotfile)
    #-B-----------------------------------------------------------------------B-

#~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
# ] Ending word_graph class
#~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~



#~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
# [ Class keyphrase_reranker
#~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
class keyphrase_reranker:
    """
    The *keyphrase_reranker* reranks a list of compression candidates according 
    to the keyphrases they contain. Keyphrases are extracted from the set of 
    related sentences using a modified version of the TextRank method 
    [mihalcea-tarau:2004:EMNLP]_. First, an undirected weighted graph is 
    constructed from the set of sentences in which *nodes* are (lowercased word, 
    POS) tuples and *edges* represent co-occurrences. The TextRank algorithm is
    then applied on the graph to assign a score to each word. Second, keyphrase
    candidates are extracted from the set of sentences using POS syntactic 
    filtering. Keyphrases are then ranked according to the words they contain.
    This class requires a set of related sentences (as a list of POS annotated 
    sentences) and the N-best compression candidates (as a list of (score, list 
    of (word, POS) tuples) tuples). The following optional parameters can be 
    specified:

    - lang is the language parameter and is used for selecting the correct 
      POS tags used for filtering keyphrase candidates.
    - patterns is a list of extra POS patterns (regexes) used for filtering 
      keyphrase candidates, default is ``^(JJ)*(NNP|NNS|NN)+$`` for English and 
      ``^(ADJ)*(NC|NPP)+(ADJ)*$`` for French.

    .. [mihalcea-tarau:2004:EMNLP] Rada Mihalcea and Paul Tarau, TextRank: 
       Bringing Order into Texts, Empirical Methods in Natural Language 
       Processing (EMNLP), 2004.
    """

    #-T-----------------------------------------------------------------------T-
    def __init__(self, sentence_list, nbest_compressions, lang="en", 
                 patterns=[], stopwords=[]):

        self.sentences = list(sentence_list)
        """ The list of related sentences provided by the user. """

        self.nbest_compressions = nbest_compressions
        """ The nbest compression candidates provided by the user. """

        self.graph = nx.Graph()
        """ The graph used for keyphrase extraction. """

        self.lang = lang
        """ The language of the input sentences, default is English (en)."""

        self.stopwords = set(stopwords) 
        """ The set of words to be excluded from keyphrase extraction. """

        self.syntactic_filter = ['JJ', 'NNP', 'NNS', 'NN', 'NNPS']
        """ The POS tags used for generating keyphrase candidates. """

        self.keyphrase_candidates = {}
        """ Keyphrase candidates generated from the set of sentences. """

        self.word_scores = {}
        """ Scores for each word computed with TextRank. """

        self.keyphrase_scores = {}
        """ Scores for each keyphrase candidate. """

        self.syntactic_patterns = ['^(JJ)*(NNP|NNS|NN)+$']
        """ Syntactic patterns for filtering keyphrase candidates. """

        # Specific rules for French
        if self.lang == "fr":
            self.syntactic_filter = ['NPP', 'NC', 'ADJ']
            self.syntactic_patterns = ['^(ADJ)*(NC|NPP)+(ADJ)*$']

        # Add extra patterns
        self.syntactic_patterns.extend(patterns)

        # 1. Build the word graph from the sentences
        self.build_graph()

        # 2. Generate the keyphrase candidates
        self.generate_candidates()

        # 3. Compute the TextRank scores for each word in the graph
        self.undirected_TextRank()

        # 4. Compute the score of each keyphrase candidate
        self.score_keyphrase_candidates()

        # 5. Cluster keyphrases to remove redundancy
        self.cluster_keyphrase_candidates()

    #-B-----------------------------------------------------------------------B-


    #-T-----------------------------------------------------------------------T-
    def build_graph(self, window=0):
        """
        Build a word graph from the list of sentences. Each node in the graph 
        represents a word. An edge is created between two nodes if they co-occur
        in a given window (default is 0, indicating the whole sentence).
        """

        # For each sentence 
        for i in range(len(self.sentences)):
        
            # Normalise extra white spaces
            self.sentences[i] = re.sub(' +', ' ', self.sentences[i])
            
            # Tokenize the current sentence in word/POS
            sentence = self.sentences[i].split(' ')

            # 1. Looping over the words and creating the nodes. Sentences are
            #    also converted to a list of tuples
            for j in range(len(sentence)):

                # Convert word/POS to (word, POS) tuple
                word, pos = self.wordpos_to_tuple(sentence[j])

                # Replace word/POS by (word, POS) tuple in the sentence
                sentence[j] = (word.lower(), pos)

                # Modify the POS tags of stopwords to exclude them
                if sentence[j][0] in self.stopwords:
                    sentence[j] = (sentence[j][0], "STOPWORD")     

                # Add the word only if it belongs to one of the syntactic 
                # categories
                if sentence[j][1] in self.syntactic_filter:

                    # Add node to the graph if not exists
                    if not self.graph.has_node(sentence[j]):
                        self.graph.add_node(sentence[j])

            # 2. Create the edges between the nodes using co-occurencies
            for j in range(len(sentence)):

                # Get the first node
                first_node = sentence[j]

                # Switch to set the window for the whole sentence
                max_window = window 
                if window < 1:
                    max_window = len(sentence)

                # For the other words in the window
                for k in range(j+1, min(len(sentence), j+max_window)):

                    # Get the second node
                    second_node = sentence[k]

                    # Check if nodes exists 
                    if self.graph.has_node(first_node) and \
                       self.graph.has_node(second_node):

                        # Add edge if not exists
                        if not self.graph.has_edge(first_node, second_node):
                            self.graph.add_edge(first_node,second_node,weight=1)
                        # Else modify weight
                        else:
                            self.graph[first_node][second_node]['weight'] += 1

            # Replace sentence by the list of tuples
            self.sentences[i] = sentence
    #-B-----------------------------------------------------------------------B-


    #-T-----------------------------------------------------------------------T-
    def generate_candidates(self):
        """
        Function to generate the keyphrase candidates from the set of related 
        sentences. Keyphrases candidates are the largest n-grams containing only
        words from the defined syntactic categories.
        """

        # For each sentence 
        for i in range(len(self.sentences)):

            sentence = self.sentences[i]

            # List for iteratively constructing a keyphrase candidate
            candidate = []

            # For each (word, pos) tuple in the sentence
            for j in range(len(sentence)):

                word, pos = sentence[j]

                # If word is to be included in a candidate
                if pos in self.syntactic_filter:

                    # Adds word to candidate
                    candidate.append(sentence[j])

                # If a candidate keyphrase is in the buffer
                elif len(candidate) > 0 and self.is_a_candidate(candidate):

                    # Add candidate
                    keyphrase = ' '.join(u[0] for u in candidate)
                    self.keyphrase_candidates[keyphrase] = candidate

                    # Flush the buffer
                    candidate = []

                else:
                    
                    # Flush the buffer
                    candidate = []
               
            # Handle the last possible candidate
            if len(candidate) > 0 and self.is_a_candidate(candidate):

                # Add candidate
                keyphrase = ' '.join(u[0] for u in candidate)
                self.keyphrase_candidates[keyphrase] = candidate
    #-B-----------------------------------------------------------------------B-


    #-T-----------------------------------------------------------------------T-
    def is_a_candidate(self, keyphrase_candidate):
        """
        Function to check if a keyphrase candidate is a valid one according to 
        the syntactic patterns.
        """

        candidate_pattern = ''.join(u[1] for u in keyphrase_candidate)

        for pattern in self.syntactic_patterns:
            if not re.search(pattern, candidate_pattern):
                return False

        return True
    #-B-----------------------------------------------------------------------B-


    #-T-----------------------------------------------------------------------T-
    def undirected_TextRank(self, d=0.85, f_conv=0.0001):
        """
        Implementation of the TextRank algorithm as described in 
        [mihalcea-tarau:2004:EMNLP]_. Node scores are computed iteratively until
        convergence (a threshold is used, default is 0.0001). The dampling 
        factor is by default set to 0.85 as recommended in the article.
        """

        # Initialise the maximum node difference for checking stability
        max_node_difference = f_conv
    
        # Initialise node scores to 1
        self.word_scores = {}
        for node in self.graph.nodes():
            self.word_scores[node] = 1.0

        # While the node scores are not stabilized
        while (max_node_difference >= f_conv):

            # Create a copy of the current node scores
            current_node_scores = self.word_scores.copy()

            # For each node I in the graph
            for node_i in self.graph.nodes():

                sum_Vj = 0

                # For each node J connected to I
                for node_j in self.graph.neighbors_iter(node_i):

                    wji = self.graph[node_j][node_i]['weight']
                    WSVj = current_node_scores[node_j]
                    sum_wjk = 0.0

                    # For each node K connected to J
                    for node_k in self.graph.neighbors_iter(node_j):
                        sum_wjk += self.graph[node_j][node_k]['weight']

                    sum_Vj += ( (wji * WSVj) / sum_wjk )

                # Modify node score
                self.word_scores[node_i] = (1 - d) + (d * sum_Vj)

                # Compute the difference between old and new score
                score_difference = math.fabs(self.word_scores[node_i] \
                                   - current_node_scores[node_i])

                max_node_difference = max(score_difference, score_difference)
    #-B-----------------------------------------------------------------------B-


    #-T-----------------------------------------------------------------------T-
    def score_keyphrase_candidates(self):
        """
        Function to compute the score of each keyphrase candidate according to 
        the words it contains. The score of each keyphrase is calculated as the 
        sum of its word scores normalized by its length + 1.
        """

        # Compute the score of each candidate according to its words
        for keyphrase in self.keyphrase_candidates:

            # Compute the sum of word scores for each candidate
            keyphrase_score = 0.0
            for word_pos_tuple in self.keyphrase_candidates[keyphrase]:
                keyphrase_score += self.word_scores[word_pos_tuple]

            # Normalise score by length
            keyphrase_score /= (len(self.keyphrase_candidates[keyphrase]) + 1.0)

            # Add score to the keyphrase candidates
            self.keyphrase_scores[keyphrase] = keyphrase_score
    #-B-----------------------------------------------------------------------B-


    #-T-----------------------------------------------------------------------T-
    def cluster_keyphrase_candidates(self):
        """
        Function to cluster keyphrase candidates and remove redundancy. A large 
        number of the generated keyphrase candidates are redundant. Some 
        keyphrases may be contained within larger ones, e.g. *giant tortoise*
        and *Pinta Island giant tortoise*. To solve this problem, generated 
        keyphrases are clustered using word overlap. For each cluster, the 
        keyphrase with the highest score is selected.
        """

        # Sort keyphrase candidates by length
        descending = sorted(self.keyphrase_candidates, 
                            key = lambda x: len(self.keyphrase_candidates[x]), 
                            reverse=True)

        # Initialize the cluster container
        clusters = {}

        # Loop over keyphrases by decreasing length
        for keyphrase in descending:

            found_cluster = False
            
            # Create a set of words from the keyphrase
            keyphrase_words = set(keyphrase.split(' '))

            # Loop over existing clusters
            for cluster in clusters:

                # Create a set of words from the cluster representative
                cluster_words = set(cluster.split(' '))

                # Check if keyphrase words are all contained in the cluster
                # representative words
                if len(keyphrase_words.difference(cluster_words)) == 0 :
                    
                    # Add keyphrase to cluster
                    clusters[cluster].append(keyphrase)

                    # Mark cluster as found
                    found_cluster = True

            # If keyphrase does not fit into any existing cluster
            if not found_cluster:
                clusters[keyphrase] = [keyphrase]

        # Initialize the best candidate cluster container
        best_candidate_keyphrases = []

        # Loop over the clusters to find the best keyphrases
        for cluster in clusters:

            # Find the best scored keyphrase candidate in the cluster
            sorted_cluster = sorted(clusters[cluster], 
                            key=lambda cluster: self.keyphrase_scores[cluster], 
                            reverse=True)

            best_candidate_keyphrases.append(sorted_cluster[0])

        # Initialize the non redundant clustered keyphrases
        non_redundant_keyphrases = []

        # Sort best candidate by score
        sorted_keyphrases = sorted(best_candidate_keyphrases, 
                        key=lambda keyphrase: self.keyphrase_scores[keyphrase], 
                        reverse=True)

        # Last loop to remove redundancy in cluster best candidates
        for keyphrase in sorted_keyphrases:
            is_redundant = False
            for prev_keyphrase in non_redundant_keyphrases:
                if keyphrase in prev_keyphrase:
                    is_redundant = True
                    break
            if not is_redundant:
                non_redundant_keyphrases.append(keyphrase)

        # Modify the keyphrase candidate dictionnaries according to the clusters
        for keyphrase in self.keyphrase_candidates.keys():

            # Remove candidate if not in cluster
            if not keyphrase in non_redundant_keyphrases:
                del self.keyphrase_candidates[keyphrase]
                del self.keyphrase_scores[keyphrase]
    #-B-----------------------------------------------------------------------B-


    #-T-----------------------------------------------------------------------T-
    def rerank_nbest_compressions(self):
        """
        Function that reranks the nbest compressions according to the keyphrases
        they contain. The cummulative score (original score) is normalized by 
        (compression length * Sum of keyphrase scores).
        """

        reranked_compressions = []

        # Loop over the compression candidates
        for cummulative_score, path in self.nbest_compressions:

            # Generate the sentence form the path
            compression = ' '.join([u[0] for u in path])

            # Initialize total keyphrase score
            total_keyphrase_score = 1.0

            # Loop over the keyphrases and sum the scores
            for keyphrase in self.keyphrase_candidates:
                if keyphrase in compression:
                    total_keyphrase_score += self.keyphrase_scores[keyphrase]

            score = ( cummulative_score / (len(path) * total_keyphrase_score) )

            bisect.insort( reranked_compressions, 
                           (score, path) )

        return reranked_compressions
    #-B-----------------------------------------------------------------------B-

    #-T-----------------------------------------------------------------------T-
    def wordpos_to_tuple(self, word, delim='/'):
        """
        This function converts a word/POS to a (word, POS) tuple. The character
        used for separating word and POS can be specified (default is /).
        """

        # Splitting word, POS using regex
        m = re.match("^(.+)"+delim+"(.+)$", word)

        # Extract the word information
        token, POS = m.group(1), m.group(2)

        # Return the tuple 
        return (token.lower(), POS)
    #-B-----------------------------------------------------------------------B-


    #-T-----------------------------------------------------------------------T-
    def tuple_to_wordpos(self, wordpos_tuple, delim='/'):
        """
        This function converts a (word, POS) tuple to word/POS. The character 
        used for separating word and POS can be specified (default is /).
        """

        # Return the word +delim+ POS
        return wordpos_tuple[0]+delim+wordpos_tuple[1]
    #-B-----------------------------------------------------------------------B-


#~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
# ] Ending keyphrase_reranker class
#~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~

