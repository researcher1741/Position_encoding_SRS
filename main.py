from src.train import *
import itertools

if __name__ == '__main__':
    import numpy as np
    activations = ["leakyrelu", "silu"]
    encodings = ["RMHA4", "carca",
                 "ROPEONE", "ROPE",
                 "conlearnt", "learnt",
                 "conRotatory", "Rotatory",
                 "con", "nocon"]
    nmaxs = [0.1, 0.0001]
    sasrecplus = True

    for activation, encoding, nmax in itertools.product(activations, encodings, nmaxs):
        print(activation, encoding, nmax)
        print(callme(activation, encoding, nmax, 2, 'fashion', 0, 6, sasrecplus))
