from src.train import *

if __name__ == '__main__':
    cases = [
    ]
    for case in cases:
        hidden_act, encoding, max_norm, norm_type, dataset, need = case
        print(callme(hidden_act, encoding, max_norm, norm_type, dataset, need))

        # ("silu", "LongRMHA4", 0.0001, 2.0, "fashion", 6),
        # ("leaky", "LongRMHA4", 0.0001, 2.0, "men", 6),
        # ("silu", "LongRMHA4", 0.0001, 2.0, "men", 6),
        # ("leaky", "LongRMHA4", 0.0001, 2.0, "fashion", 6),
        # ("leaky", "ROPEMHA", None, 2.0, "games", 6),
        # ("silu", "ROPEMHA", None, 2.0, "games", 6),
        # ("leaky", "RMHA4", None, 2.0, "games", 6),
        # ("silu", "RMHA4", None, 2.0, "games", 6),
        # ("leaky", "conlearnt", None, 2.0, "games", 6),
        # ("silu", "conlearnt", None, 2.0, "games", 6),
        # ("leaky", "learnt", None, 2.0, "games", 6),
        # ("silu", "learnt", None, 2.0, "games", 6),
        # ("leaky", "carca", None, 2.0, "games", 6),
        # ("silu", "carca", None, 2.0, "games", 6),
        # ("leaky", "con", None, 2.0, "games", 6),
        # ("silu", "con", None, 2.0, "games", 6),
        # ("leaky", "nocon", None, 2.0, "games", 6),
        # ("silu", "nocon", None, 2.0, "games", 6),
        # ("leaky", "conROPE", None, 2.0, "games", 6),
        # ("silu", "conROPE", None, 2.0, "games", 6),
        # ("leaky", "ROPE", None, 2.0, "games", 6),
        # ("silu", "ROPE", None, 2.0, "games", 6),
        # ("leaky", "ROPEMHAONE", None, 2.0, "games", 6),
        # ("silu", "ROPEMHAONE", None, 2.0, "games", 6),
        # ("leaky", "ROPEMHAONE", 0.0001, 2.0, "fashion", 6),
        # ("silu", "ROPEMHAONE", 0.0001, 2.0, "fashion", 6),
        # ("leaky", "ROPEMHAONE", 0.0001, 2.0, "men", 6),
        # ("silu", "ROPEMHAONE", 0.0001, 2.0, "men", 6),
        # ("leaky", "ROPEMHAONE", 0.0001, 2.0, "beauty", 3),
        # ("silu", "ROPEMHAONE", 0.0001, 2.0, "beauty", 3),
        # ("leaky", "LongconROPE", None, 2.0, "games", 6),
        # ("silu", "LongconROPE", None, 2.0, "games", 6),
        # ("leaky", "LongROPE", None, 2.0, "games", 6),
        # ("silu", "LongROPE", None, 2.0, "games", 6),
