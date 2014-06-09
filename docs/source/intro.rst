Introduction
============

Potara is a multi-document summarization system that relies on Integer
Linear Programming (ILP) and sentence fusion.

Its goal is to summarize a set of related documents.
It proceeds by fusing similar sentences in order to create sentence
that are either shorter or more informative than those found in the
documents.
It then uses ILP in order to choose the best set of sentences, fused
or not, that will compose the resulting summary.

It relies on state-of-the-art approaches introduced by :ref:`Gillick and
Favre <gillandfavre>` for the ILP strategy, and :ref:`Filipppova <filippova>` for the sentence fusion.


Citations
---------

.. _gillandfavre:

    Gillick and Favre : A scalable gobal model for summarization, 2009

.. _filippova:

    Katja Filippova : Multi-sentence compression: Finding shortest paths in word graphs, 2010
