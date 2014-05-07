from complearning import GEN, sem

# This lexicon is not involved in the algorithms, but it can be used
# to create train/test data, via a string s supplied to GEN with the
# signature GEN(s, lexicon=gold_lexicon).
gold_lexicon = {
    'one': ['1'],
    'two': ['2'],
    'three': ['3'],
    'four': ['4'],
    'five': ['5'],
    'six': ['6'],
    'seven': ['7'],
    'eight': ['8'],
    'nine': ['9'],
    'plus': ['+'],
    'minus': ['-', '~'],
    'times': ['*'],
}

# Train/test data for the demo:
utterances = [
    # TRAIN:
    'one plus two',
    'two plus two',
    'two plus three',
    'three plus one',
    'three plus minus two',
    'two plus two',
    'three minus two',
    'minus three minus two',
    'two times two',
    'one times one',
    'two times three',
    'three plus three minus two',
    # TEST:
    'minus three',
    'three plus two',
    'two times two plus three',
    'minus four'
]

for u in utterances:
    lf = GEN(u, lexicon=gold_lexicon)[0]    
    x = [u, lf, sem(lf)]
    print x
    
