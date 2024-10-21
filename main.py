from src.train import *

if __name__ == '__main__':
    import numpy as np
    cases = [
            {'activation': 'leaky', 'encoding': 'carca', 'nmax': 0.0001, 'ntype': 2.0},
            {'activation': 'leaky', 'encoding': 'conRotatory', 'nmax': 0.0001, 'ntype': 2.0},
            {'activation': 'leaky', 'encoding': 'RMHA4', 'nmax': 0.0001, 'ntype': 2.0},
            {'activation': 'leaky', 'encoding': 'carca', 'nmax': 0.1, 'ntype': 2.0},
            {'activation': 'leaky', 'encoding': 'conRotatory', 'nmax': 0.1, 'ntype': 2.0},
            {'activation': 'leaky', 'encoding': 'RMHA4', 'nmax': 0.1, 'ntype': 2.0},
    ]
    for case in cases:
        print(callme(case['activation'], case['encoding'], case['nmax'], case['ntype'], 'submen3', 0, 6))
