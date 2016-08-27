#!/usr/bin/env python

"""
Generates the dataset used for training and evaluating the semantic
model defined in `synthesis.py`. The basis for this is `gold_lexicon`
and its grammar, as defined in `grammar.py`.
"""

__author__ = "Christopher Potts and Percy Liang"
__copyright__ = "Copyright 2014-, Christopher Potts and Percy Liang"
__credits__ = []
__license__ = "GNU general public license, version 2"
__version__ = "2.0"
__maintainer__ = "Christopher Potts"
__email__ = "See the authors' websites"


from grammar import *


# Gold grammar for creating the data:
gram = Grammar(gold_lexicon, rules, functions)

# Train/test data for the demo:
train_utterances = [
    'one plus one',
    'one plus two',    
    'one plus three',
    'two plus two',    
    'two plus three',
    'three plus one',
    'three plus minus two',
    'two plus two',
    'three minus two',
    'minus three minus two',
    'two times two',
    'two times three',
    'three plus three minus two'
]

# Test:
test_utterances = [
    'minus three',
    'three plus two',
    'two times two plus three',
    'minus four'
]

# This is a list of triples (x, y, d), where x is an input string, y
# is its preferred logical form, and d is the denotation of the
# logical form y. The semantic parsing algorithms use only x and y,
# and the interpretive algorithms use only x and d.
sem_train = []
for u in train_utterances:
    # Pick the first in the list of possible LFs, since this happens
    # to correspond to the preferences we expressed in the paper:
    lf = gram.gen(u)[0] 
    sem_train.append([u, lf, gram.sem(lf)])

# A list of triples with the same format as the training data; the
# algorithms should do well on the first three and fail on the last
# one because 'four'/4 never appears in the training data.
sem_test = []
for u in test_utterances:
    # Pick the last in the list of possible LFs, since this happens to 
    # correspond to the preferences we expressed in the paper:
    lf = gram.gen(u)[-1]
    sem_test.append([u, lf, gram.sem(lf)])

