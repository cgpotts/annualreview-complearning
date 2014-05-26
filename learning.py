#!/usr/bin/env python

"""
Defines the core learning framework.

The framework defined by score, predict, and SGD is defined in section
3.2 of the paper. See evenodd.py for a simple example (corresponding
to table 3).

This core framework is also all that is needed for simple semantic
parsing: section 4.1 of the paper and evaluate_semparse in
synthesis.py.

For learning from denotations (section 4.2 of the paper), the
framework is defined by score, predict, and LatentSGD. See
evaluate_interpretive in synthesis.py

We don't cover this in the paper, but score, predict, and LatentSGD
can also be used for semantic parsing where the full tree structure of
the logical form is hidden, and only the root node logical expression
is available for training. See evaluate_latent_semparse in
synthesis.py

The function evaluate below provides a generic interface for showing
basic results for train/test sets.
"""

__author__ = "Christopher Potts and Percy Liang"
__copyright__ = "Copyright 2014, Christopher Potts and Percy Liang"
__credits__ = []
__license__ = "GNU general public license, version 2"
__version__ = "0.1"
__maintainer__ = "Christopher Potts"
__email__ = "See the authors' websites"

import re
import random
from collections import defaultdict
from operator import itemgetter
from itertools import product

def score(x=None, y=None, phi=None, w=None):
    """Calculates the inner product w * phi(x,y)."""
    return sum(w[f]*count for f, count in phi(x, y).items())

def predict(x=None, w=None, phi=None, classes=None, output_transform=(lambda x : x)):    
    scores = [(score(x, y_prime, phi, w), y_prime) for y_prime in classes(x)]
    # Get the maximal score:
    max_score = sorted(scores)[-1][0]
    # Get all the candidates with the max score and choose one randomly:
    y_hats = [y_alt for s, y_alt in scores if s == max_score]
    return output_transform(random.choice(y_hats))

######################################################################
# Note: SGD and LatentSGD can be seen as differing only in how they
# choose the hidden variable y: for SGD, it is the same as the output
# seen in the training data, whereas LatentSGD chooses it as the
# highest scoring hidden variable. Thus, SGD and LatentSGD could be
# stated as abstractions of a single function, call it GenericSGD,
# differing only in the function used to choose this value: an
# identity function for SGD and the best prediction for LatentSGD (see
# the first line of the loop through the training data). We have not
# combined them here in order to keep the code readable, but combining
# could help bring out this insight (and make for more maintainable
# code).
######################################################################

def SGD(D=None, phi=None, classes=None, T=10, eta=0.1, output_transform=None):
    """Implements stochatic (sub)gradient descent, as in the paper. classes should be a
    function of the input x for structure prediction cases (where classes is GEN)."""
    w = defaultdict(float)
    for t in range(T):
        random.shuffle(D)
        for x, y in D:
            # Get all (score, y') pairs:
            scores = [(score(x, y_alt, phi, w)+cost(y, y_alt), y_alt) for y_alt in classes(x)]
            # Get the maximal score:
            max_score = sorted(scores)[-1][0]
            # Get all the candidates with the max score and chose one randomly:
            y_tildes = [y_alt for s, y_alt in scores if s == max_score]
            y_tilde = random.choice(y_tildes)
            # Weight-update (a bit cumbersome because of the dict-based implementation):
            actual_rep = phi(x, y)
            predicted_rep = phi(x, y_tilde)
            for f in set(actual_rep.keys() + predicted_rep.keys()):
                w[f] += eta * (actual_rep[f] - predicted_rep[f])                             
    return w

def LatentSGD(D=None, phi=None, classes=None, T=10, eta=0.1, output_transform=None):
    """Implements stochatic (sub)gradient descent for the latent SVM
    objective, as in the paper. classes is defined as GEN(x, d) for
    each input x."""
    w = defaultdict(float)
    for t in range(T):
        random.shuffle(D)
        for x, d in D:
            # Get the best viable candidate given the current weights:
            y = predict(x, w, phi=phi, classes=(lambda z : [zd for zd in classes(z) if output_transform(zd) == d]))
            # Get all (score, y') pairs:
            scores = [(score(x, y_alt, phi, w)+cost(y, y_alt), y_alt) for y_alt in classes(x)]
            # Get the maximal score:
            max_score = sorted(scores)[-1][0]
            # Get all the candidates with the max score and chose one randomly:
            y_tildes = [y_alt for s, y_alt in scores if s == max_score]
            y_tilde = random.choice(y_tildes)
            # Weight-update:
            actual_rep = phi(x, y)
            predicted_rep = phi(x, y_tilde)
            for f in set(actual_rep.keys() + predicted_rep.keys()):
                w[f] += eta * (actual_rep[f] - predicted_rep[f])            
    return w

def cost(y, y_prime):
    """Cost function used by SGD (above) and LatentSGD (below)."""
    return 0.0 if y == y_prime else 1.0

def evaluate(phi=None, 
             optimizer=None, 
             train=None, 
             test=None, 
             classes=None, 
             T=10, 
             eta=0.1, 
             output_transform=(lambda x : x)):
    """Generic interface for showing learning weights and train/test 
    results. optimizer should be SGD or LatentSGD, classes should be 
    a function of the inputs x, and output_tranform is used only by 
    models with latent variables. For examples of use, see evenodd.py 
    and synthesis.py."""
    print "======================================================================"    
    print "Feature function: " + phi.__name__    
    w = optimizer(D=train, phi=phi, T=T, eta=eta, classes=classes, output_transform=output_transform)
    print "--------------------------------------------------"
    print 'Learned feature weights'
    for f, val in sorted(w.items(), key=itemgetter(1), reverse=True):
        print f, val
    for label, data in (('Train', train), ('Test', test)):
        print "--------------------------------------------------"
        print '%s predictions' % label
        for x, y in data:
            prediction = predict(x, w, phi=phi, classes=classes, output_transform=output_transform)
            print '\nInput:   ', x
            print 'Gold:      ', y
            print 'Prediction:', prediction
            print 'Correct:   ', y == prediction
