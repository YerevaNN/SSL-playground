#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 8/12/20
import numpy as np


def select_uniform(train_sents, test_types, max_budget):
    """
    train_lines: are list of tokenized lines
    test_types: are list of types
    """
    assert isinstance(train_sents, list)
    assert isinstance(train_sents[0], list)
    assert isinstance(train_sents[0][0], str)
    assert isinstance(test_types, list)
    assert isinstance(test_types[0], str)

    oov = '<<other>>'
    costs = np.array([len(' '.join(line)) for line in train_sents])
    idx2str = [oov] + test_types
    str2idx = {t: idx for idx, t in enumerate(idx2str)}
    oov_idx = str2idx[oov]

    cands = []
    for line in train_sents:
        bag = set([str2idx.get(tok, oov_idx) for tok in line])
        if oov_idx in bag:
            bag.remove(oov_idx)  # exclude oov
        cands.append(bag)

    assert len(cands) == len(costs)
    coverage = np.array([len(bag) for bag in cands])
    worths = coverage / costs

    return select_max_worth(costs, worths, max_budget)


def select_rare(train_sents, test_tfs, max_budget):
    """
    train_lines: are list of tokenized lines
    test_types: are list of types
    """
    assert isinstance(train_sents, list)
    assert isinstance(train_sents[0], list)
    assert isinstance(train_sents[0][0], str)
    assert isinstance(test_tfs, dict)

    costs = np.array([len(' '.join(line)) for line in train_sents])

    total = sum(test_tfs.values())
    inv_freq = {t: total / f for t, f in test_tfs.items()}
    value_per_sent = [sum(inv_freq.get(tok, 0) for tok in cand) for cand in train_sents]
    value_per_sent = np.array(value_per_sent)

    assert len(value_per_sent) == len(costs)
    worths = value_per_sent / costs
    return select_max_worth(costs, worths, max_budget)


def select_max_worth(costs, worths, max_budget):
    # descending order of worths
    worth_idx = reversed(np.argsort(worths, axis=0))
    selection = np.zeros(len(worths), dtype=np.int8)  # nothing selected; all zeros
    cur_budget = 0
    for idx in worth_idx:
        if cur_budget + costs[idx] <= max_budget:
            selection[idx] = 1
            cur_budget += costs[idx]
    return selection
