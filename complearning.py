#!/usr/bin/env python

"""
Demonstration code for the paper

Liang, Percy and Christopher Potts. 2014. Bringing machine learning
and compositional semantics together. Submitted to the Annual
Review of Linguistics.

The purpose of the code is just to illustrate how the algorithms work,
as an aid to understanding the paper and developing new models
that synthesize compositionality and machine learning.

"""

__author__ = "Christopher Potts"
__copyright__ = "Copyright 2014, Christopher Potts"
__credits__ = []
__license__ = "MIT License (MIT)"
__version__ = "0.1"
__maintainer__ = "Christopher Potts"
__email__ = "See the author's website"

import re
from collections import defaultdict
from operator import itemgetter
from itertools import product
from random import shuffle

######################################################################
## SCORING AND PREDICTION

def score(x=None, y=None, phi=None, w=None):
    """Calculates the inner product w * phi(x,y)."""
    return sum(w[f]*count for f, count in phi(x, y).items())

def predict(x=None, w=None, phi=None, classes=None, transform=(lambda x : x)):    
    scores = [(score(x, y_prime, phi, w), y_prime) for y_prime in classes(x)]
    return transform(sorted(scores)[-1][1])

######################################################################
## OPTIMIZATION

def SGD(D=None, phi=None, classes=None, T=1000, eta=0.03):
    """Implements stochatic (sub)gradient descent, as in the paper. classes should be a
    function of the input x for structure prediction cases (where classes is GEN)."""
    w = defaultdict(float)
    for t in range(T):
        shuffle(D)
        for x, y in D:
            # Get all (score, y') pairs:
            scores = [(score(x, y_alt, phi, w)+cost(y, y_alt), y_alt) for y_alt in classes(x)]
            # The argmax is the highest scoring label (bottom of the list):
            y_tilde = sorted(scores)[-1][1]
            # Weight-update (a bit cumbersome because of the dict-based implementation):
            actual_rep = phi(x, y)
            predicted_rep = phi(x, y_tilde)
            for f in set(actual_rep.keys() + predicted_rep.keys()):
                w[f] += eta * (actual_rep[f] - predicted_rep[f])
    return w

def cost(y, y_prime):
    """Cost function used by SGD (above) and LatentSGD (below)."""
    return 0.0 if y == y_prime else 1.0

######################################################################
## EVEN/ODD FEATURE FUNCTIONS

def phi_empty_string(x, y):
    """Feature function that tracks only the class bias."""
    d = defaultdict(float)
    d[y] = 1.0
    return d

def phi_last_word(x, y):
    """Feature function that tracks the class bias and the last word."""
    s = x.rsplit(' ', 1)[1]
    d = defaultdict(float)
    d[y] = 1.0
    d[(s,y)] = 1.0
    return d

def phi_all_words(x, y):
    """Feature function that tracks the class bias and all the words."""
    strs = x.split()
    d = defaultdict(float)
    d[y] = 1.0
    for s in strs:
        d[(s,y)] = 1.0
    return d

######################################################################
## GEN AND ITS INTERPRETATION

def GEN(x=None, lexicon=None):
    """Given a lexicon, produce all the logical forms for input_str
    given the grammar defined by rules. If d is specified, filter off
    LFs whose denotion is not d."""
    lfs = parse(x.split(), lexicon=lexicon)        
    return lfs
          
def parse(words, lexicon=None):
    """CYK parsing, which we can get away with because it is
    easy to define regular expressions over our logical forms,
    thereby allowing us to use a standard chart parsing set-up.
    The algorithm returns just the full parses."""    
    n = len(words)+1
    trace = defaultdict(list)
    for i in range(1,n):
        word = words[i-1]
        trace[(i-1,i)] = [[lex, word] for lex in lexicon[word]]
    for j in range(2, n):
        for i in range(j-1, -1, -1):
            for k in range(i+1, j):
                for c1, c2 in product(trace[(i,k)], trace[(k,j)]):
                    for m in rules(c1[0], c2[0]):                                                  
                        trace[(i,j)].append([m, c1, c2])
    return trace[(0,n-1)]

def rules(c1, c2):
    """Rules of combination. c1 and c2 are strings representing
    parts of logical forms. The only change from the paper is
    that the relational nodes are binarized for the sake of the
    parsing algorithm. The new node is labeled B."""
    results = []
    predicate_re = re.compile(r"^~$")
    intermediate_re = re.compile(r"^[*\-+] ")
    relation_re = re.compile(r"^[*\-+]$")
    num_re = re.compile(r"^~.+|^\(.+\)$|^[0-9]+$")  
    # B -> N R    
    if num_re.search(c1) and relation_re.search(c2):
        results.append('%s %s' % (c2, c1))
    # N -> B N
    if intermediate_re.search(c1) and num_re.search(c2):
        results.append('(%s %s)' % (c1, c2))
    # N -> U N
    if predicate_re.search(c1) and num_re.search(c2):
        results.append('%s%s' % (c1, c2))
    return results

def sem(lf):
    """Takes a full tree structure lf and interprets its root node
    via a quick, hacky conversion into Python code."""
    lf = lf[0] # Interpret just the root node.
    plus = (lambda x,y : x + y)
    times = (lambda x,y : x * y)
    minus = (lambda x,y : x - y)
    lf = lf.replace('(+ ', 'plus(')
    lf = lf.replace('(- ', 'minus(')
    lf = lf.replace('(* ', 'times(')
    lf = lf.replace('~', '-')
    lf = lf.replace(' ', ', ')
    return eval(lf)

######################################################################
## FEATURE FUNCTION FOR BOTH COMPOSITIONAL LEARNING THEORIES
    
def phi_semparse(x, y):
    d = defaultdict(float)
    # Topmost relation symbol:
    if len(y[0]) > 2 and y[0][1] in ('*', '+', '-'):
        d['top:' + y[0][1]] = 1.0
    # Lexical features:
    for leaf in leaves(y):
        d[leaf[0] + ':' + leaf[1]] += 1.0
    return d

def leaves(x):
    # Leaf-only trees:
    if len(x) == 2 and isinstance(x[0], str) and isinstance(x[1], str):
        return [tuple(x)]
    # Recursive call for all child subtrees:
    l = []
    for child in x[1: ]:
        l += leaves(child)
    return l
            
######################################################################
## INTERPRETIVE SGD

def LatentSGD(D=None, phi=None, classes=None, T=100, eta=0.01):
    """Implements stochatic (sub)gradient descent for the latent SVM
    objective, as in the paper. classes is defined as GEN(x, d) for
    each input x."""
    w = defaultdict(float)
    for t in range(T):
        shuffle(D)
        for x, d in D:
            # Get the best viable candidate given the current weights:
            y = predict(x, w, phi=phi, classes=(lambda z : [zd for zd in classes(z) if sem(zd) == d]))
            # Get all (score, y') pairs:
            scores = [(score(x, y_alt, phi, w)+cost(y, y_alt), y_alt) for y_alt in classes(x)]
            # The argmax is the highest scoring label (bottom of the list):
            y_tilde = sorted(scores)[-1][1]
            # Weight-update:
            actual_rep = phi(x, y)
            predicted_rep = phi(x, y_tilde)
            for f in set(actual_rep.keys() + predicted_rep.keys()):
                w[f] += eta * (actual_rep[f] - predicted_rep[f])
    return w

######################################################################
## GENERIC EVALUATION FUNCTION
            
def evaluate(phi=None, optimizer=None, train=None, test=None, classes=None, T=100, eta=0.1, output_transform=(lambda x : x)):
    print "======================================================================"    
    print "FEATURE FUNCTION: " + phi.__name__    
    w = optimizer(D=train, phi=phi, T=T, eta=eta, classes=classes)
    print "--------------------------------------------------"
    print 'LEARNED FEATURE WEIGHTS'
    for f, val in sorted(w.items(), key=itemgetter(1), reverse=True):
        print f, val
    for label, data in (('TRAIN', train), ('TEST', test)):
        print "--------------------------------------------------"
        print '%s PREDICTIONS' % label
        for x, y in data:
            prediction = predict(x, w, phi=phi, classes=classes, transform=output_transform)
            print '\nInput:   ', x
            print 'Gold:      ', y
            print 'Prediction:', prediction
            print 'Correct:   ', y == prediction

######################################################################

if __name__ == '__main__':

    ##################################################################

    EVEN = 'EVEN'
    ODD = 'ODD'

    evenodd_train = [
        ['twenty five', ODD],
        ['thirty one', ODD],
        ['forty nine', ODD],
        ['fifty two', EVEN],
        ['eighty two', EVEN],
        ['eighty four', EVEN],
        ['eighty six', EVEN]
    ]

    evenodd_test = [('eighty five', ODD)]

    def evaluate_evenodd():
        for phi in (phi_empty_string, phi_last_word, phi_all_words):
            evaluate(phi=phi,
                     optimizer=SGD,
                     train=evenodd_train,
                     test=evenodd_test,
                     classes=(lambda x : (EVEN, ODD)),
                     T=1000,
                     eta=0.03)

    ##################################################################
   
    # This crude lexicon is the starting point for learning; it respects
    # typing but nothing else:
    crude_lexicon = {}
    crude_lexicon = {}
    for word in ('one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'):
        crude_lexicon[word] = [str(i) for i in range(1,10)]
    for word in ('times', 'plus', 'minus'):
        crude_lexicon[word] = ['+', '-', '*']
    crude_lexicon['minus'] += ['~']

    # This is a list of triples (x, y, d), where x is an input string,
    # y is its preferred logical form, and d is the denotation of the
    # logical form y.  The semantic parsing algorithms use only x and y,
    # and the interpretive algorithms use only x and d.
    sem_train = [
        ['one plus two', ['(+ 1 2)', ['+ 1', ['1', 'one'], ['+', 'plus']], ['2', 'two']], 3],
        ['two plus two', ['(+ 2 2)', ['+ 2', ['2', 'two'], ['+', 'plus']], ['2', 'two']], 4],
        ['two plus three', ['(+ 2 3)', ['+ 2', ['2', 'two'], ['+', 'plus']], ['3', 'three']], 5],
        ['three plus one', ['(+ 3 1)', ['+ 3', ['3', 'three'], ['+', 'plus']], ['1', 'one']], 4],
        ['three plus minus two', ['(+ 3 ~2)', ['+ 3', ['3', 'three'], ['+', 'plus']], ['~2', ['~', 'minus'], ['2', 'two']]], 1],
        ['two plus two', ['(+ 2 2)', ['+ 2', ['2', 'two'], ['+', 'plus']], ['2', 'two']], 4],
        ['three minus two', ['(- 3 2)', ['- 3', ['3', 'three'], ['-', 'minus']], ['2', 'two']], 1],
        ['minus three minus two', ['~(- 3 2)', ['~', 'minus'], ['(- 3 2)', ['- 3', ['3', 'three'], ['-', 'minus']], ['2', 'two']]], -1],
        ['two times two', ['(* 2 2)', ['* 2', ['2', 'two'], ['*', 'times']], ['2', 'two']], 4],
        ['one times one', ['(* 1 1)', ['* 1', ['1', 'one'], ['*', 'times']], ['1', 'one']], 1],
        ['two times three', ['(* 2 3)', ['* 2', ['2', 'two'], ['*', 'times']], ['3', 'three']], 6],
        ['three plus three minus two', ['(+ 3 (- 3 2))', ['+ 3', ['3', 'three'], ['+', 'plus']], ['(- 3 2)', ['- 3', ['3', 'three'], ['-', 'minus']], ['2', 'two']]], 4]
    ]

    # A list of triples with the same format as the training data; the
    # algorithms should do well on the first three and fail on the last
    # one because 'four'/4 never appears in the training data.
    sem_test = [
        ['minus three', ['~3', ['~', 'minus'], ['3', 'three']], -3],
        ['three plus two', ['(+ 3 2)', ['+ 3', ['3', 'three'], ['+', 'plus']], ['2', 'two']], 5],
        ['two times two plus three', ['(+ (* 2 2) 3)', ['+ (* 2 2)', ['(* 2 2)', ['* 2', ['2', 'two'], ['*', 'times']], ['2', 'two']], ['+', 'plus']], ['3', 'three']], 7],
        ['minus four', ['~4', ['~', 'minus'], ['4', 'four']], -4]
    ]
            
    def evaluate_semparse():
        semparse_train = [[x,y] for x, y, d in sem_train]
        semparse_test = [[x,y] for x, y, d in sem_test]        
        evaluate(phi=phi_semparse,
                 optimizer=SGD,
                 train=semparse_train,
                 test=semparse_test,
                 classes=(lambda x : GEN(x, lexicon=crude_lexicon)),
                 T=100,
                 eta=0.1)

    def evaluate_interpretive():
        interpretive_train = [[x,d] for x, y, d in sem_train]
        interpretive_test = [[x,d] for x, y, d in sem_test]
        evaluate(phi=phi_semparse,
                 optimizer=LatentSGD,
                 train=interpretive_train,
                 test=interpretive_test,
                 classes=(lambda x : GEN(x, lexicon=crude_lexicon)),
                 T=100,
                 eta=0.1,
                 output_transform=sem)


    #evaluate_evenodd()
    #evaluate_semparse()
    evaluate_interpretive()

