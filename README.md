[![Build Status](https://travis-ci.org/sildar/potara.svg?branch=master)](https://travis-ci.org/sildar/potara)
[![Coverage Status](https://coveralls.io/repos/sildar/potara/badge.png?branch=master)](https://coveralls.io/r/sildar/potara?branch=master)

# Basics

Potara is a multi-document summarization system that relies on Integer
Linear Programming (ILP) and sentence fusion.

Its goal is to summarize a set of related documents.
It proceeds by fusing similar sentences in order to create sentence
that are either shorter or more informative than those found in the
documents.
It then uses ILP in order to choose the best set of sentences, fused
or not, that will compose the resulting summary.

It relies on state-of-the-art approaches introduced by Gillick and
Favre for the ILP strategy, and Filippova for the sentence fusion.

# Install

You can install most python dependencies with pip

```
$ pip install -r requirements.txt
```

You will also need GLPK, which is used to obtain an optimal summary
(example for Debian-based distro)

```
$ sudo apt-get install glpk
```

You may also need to install scipy and numpy with your distro package
manager

```
$ sudo apt-get install python-numpy python-scipy
```

You can check that the install run successfully by running

```
$ python setup.py test
```

# How To

Basically, you can use the following

```
from summarizer import Summarizer
import document

s = Summarizer()
print("Adding docs")
s.setDocuments([document.Document('data/' + str(i) + '.txt')
       for i in range(1,10)])
print("summarizing")
s.summarize()
print(s.summary)
```

There's some preprocessing involved and a sentence fusion step, but I
made it easily tunable. Preprocessing may take a while (a few minutes)
since there is a lot going on under the hood.
