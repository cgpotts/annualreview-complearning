annualreview-complearning
=========================

Demonstration code for the paper

Liang, Percy and Christopher Potts. 2014. Bringing machine learning
and compositional semantics together. Submitted to the Annual
Review of Linguistics.

The purpose of the code is just to illustrate how the algorithms work,
as an aid to understanding the paper and developing new models
that synthesize compositionality and machine learning.

All of the files contain detailed explanations and documentation, with
cross references to the paper. evenodd.py, grammar.py, synthesis.py
run demos corresponding to examples and discussions from the paper.

## grammar.py 

Implements a simple interpreted context-free grammar formalism in
which each nonterminal node is a tuple (s, r) where s is the syntactic
category and r is the logical form representation. The user supplies a
lexicon, a rule-set, and possibly a set of functions that, together
with Python itself, make the logical forms interpretable as Python
code.

## semdata.py

Uses grammar.py to create training and testing data for the semantic
models.

## learning.py

The core learning framework from section 3.2 of the paper.

## evenodd.py

A simple supervised learning example using learning.py. The examples
correspond to table 3 in the paper.

## synthesis.py

Implements three different theories for learning compositional
semantics. All are illustrated with the same feature function and
train/test data.

* Basic semantic parsing, in which we learn from and predict entire
tree-structure logical forms.  This involves no latent variables;
optimization is with SGD. This is the focus of section 4.1 of the
paper.

* Learning from denotations, in which the tree-structural logical
forms are latent variables; optimization is with LatentSGD. This
is the focus of section 4.2 of the paper.

* Latent variable semantic parsing, in which we learn from and predict
only the root node of the logical form, making the tree structure a
hidden variable. This is not covered in detail in the paper due to
space constraints, but it achieves a richer connection with the
literature on semantic parsing.  To make things interesting, we add a
type-lifting rule to the grammar for this example, so that individual
logical forms correspond to multiple derivations.

