[![Build Status](https://travis-ci.org/sildar/potara.svg?branch=master)](https://travis-ci.org/sildar/potara)
[![Coverage Status](https://coveralls.io/repos/sildar/potara/badge.png?branch=master)](https://coveralls.io/r/sildar/potara?branch=master)
[![Requirements Status](https://requires.io/github/sildar/potara/requirements.svg?branch=master)](https://requires.io/github/sildar/potara/requirements/?branch=master)

# Basics

Potara is a **multi-document** summarization system that relies on Integer
Linear Programming (ILP) and sentence fusion.

Its goal is to summarize a set of related documents in a few sentences.
It proceeds by fusing similar sentences in order to create sentences
that are either shorter or more informative than those found in the
documents.
It then uses ILP in order to choose the best set of sentences, fused
or not, that will compose the resulting summary.

It relies on state-of-the-art (as of 2014) approaches introduced by Gillick and
Favre for the ILP strategy, and Filippova for the sentence fusion.

# Install

## The easy way

You should be able to install potara and its dependencies with pip

```
pip install potara
```

You can also clone this repo and use the requirements.txt file to install dependencies

## further requirements

You will also need GLPK, which is used to obtain an optimal summary
(example for Debian-based distro)

```
$ sudo apt-get install glpk
```

For Ubuntu-based distros you can use:
```
$ sudo apt-get install libglpk40
```

You can check that the install run successfully by cloning the repo and running

```
$ python setup.py test
```

If you have issues with install, you can check the .travis.yml file of the repo, which corresponds to a working build.

# How To

Basically, you can use the following

```
from summarizer import Summarizer
import document

s = Summarizer()

# Adding docs, preprocessing them and computing some infos for the summarizer
s.setDocuments([document.Document('data/' + str(i) + '.txt')
                for i in range(1,10)])
       
# Summarizing, where the actual work is done
s.summarize()

# You can then print the summary
print(s.summary)
```

There's some preprocessing involved and a sentence fusion step, but I
made it easily tunable. Preprocessing may take a while (a few minutes)
since there is a lot going on under the hood. Default parameters are 
currently set for summarizing ~10 documents. You can summarize a smaller
amount of documents by tweaking the "minbigramcount" parameter of the
summarizer :

`s = Summarizer(minbigramcount=2)`

Summarizing less than 4 documents would probably yield a bad
summary.
