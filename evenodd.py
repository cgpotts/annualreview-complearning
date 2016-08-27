#!/usr/bin/env python

"""
Implements the simple supervised learning example from table 3 of the
paper. Use

python evenodd.py

to run the demo.

evenodd_train and evenodd_test are the data sets from column 1.

The functions `phi_empty_string`, `phi_last_word`, and `phi_all_words`
implement the feature functions in columns 2-4.  Each of them defines a
class feature.

The feature keys are (string, classname) pairs. For example, if 'five'
is seen, then there are feature keys ('five', EVEN) and ('five', ODD).
This reflects the details of our SGD implementation (in `learning.py`),
which is geared towards structure prediction.

`evaluate_evenodd` evaluates all of these feature functions using the
general learning framework in learning.py.

We expect `phi_empty_string` and `phi_all_words` to fail on the
single-example test case: `phi_empty_string` uses only the class
feature, which is biased toward `EVEN`, and `phi_all_words` doesn't
have enough of the right kind of training data to make the correct
prediction. `phi_last_word` is the best representation of the three,
given `evenodd_train`.
"""


__author__ = "Christopher Potts and Percy Liang"
__credits__ = []
__license__ = "GNU general public license, version 2"
__version__ = "2.0"
__maintainer__ = "Christopher Potts"
__email__ = "See the authors' websites"


from collections import defaultdict
from learning import evaluate, SGD


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

evenodd_test = [
    ['eighty five', ODD]
]

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

def evaluate_evenodd():
    """Evaluates all three feature functions using the generic
    evaluation interface in learning.py."""
    print("======================================================================")
    print('EVEN/ODD')
    for phi in (phi_empty_string, phi_last_word, phi_all_words):
        evaluate(phi=phi,
                 optimizer=SGD,
                 train=evenodd_train,
                 test=evenodd_test,
                 # Artificially a function of input x to handle the
                 # structure prediction cases, where classes is GEN:
                 classes=(lambda x : (EVEN, ODD)), 
                 T=10,
                 eta=0.1)


if __name__ == '__main__':
    
    evaluate_evenodd()
    
