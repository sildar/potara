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

# How To

Basically, you can use the following

```
s = Summarizer()
print("Adding docs")
s.setDocuments([document.Document('pathtofilenumber' + n)
       for i in range(1,11)])
print("summarizing")
s.summarize()
print(s.summary())
```

There's some preprocessing involved and a sentence fusion step, but I
made it easily tunable.
