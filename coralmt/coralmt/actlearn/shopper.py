#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu]
# Created: 8/7/20

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple
from coralmt.utils import term_freqs
import abc
import random


@dataclass
class LabelShopper(abc.ABC):
    """Parent of all shoppers. See derived classes for different kinds of shoppers"""
    train_df: pd.DataFrame
    test_df: pd.DataFrame

    def __post_init__(self):
        for col_name in ['id', 'source', 'source_tok', 'bought', 'cost', 'target']:
            assert col_name in self.train_df.columns
        for col_name in ['id', 'source', 'source_tok']:
            assert col_name in self.test_df.columns

    def wishlist(self, max_budget: float) -> List[Tuple[str, str]]:
        """
        :param max_budget:
        :return: wishlist of (id, source) to be shopped
        """
        raise NotImplementedError()

    def commit(self, labels: Dict[str, str]):
        """
        updates the state my marking given (id, text) as purchased
        :param labels: map of {id: ref}
        :return:
        """
        for _id, ref in labels.items():
            self.train_df.loc[self.train_df['id'] == _id, 'bought'] = True
            self.train_df.loc[self.train_df['id'] == _id, 'target'] = ref

    @classmethod
    def pick_max_worth(cls, costs, worths, max_budget: float):
        # descending order of worths
        worth_idx = reversed(np.argsort(worths, axis=0))
        selection = np.zeros(len(worths), dtype=np.int8)  # nothing selected; all zeros
        cur_budget = 0
        for idx in worth_idx:
            if cur_budget + costs[idx] <= max_budget:
                selection[idx] = 1
                cur_budget += costs[idx]
        return selection


@dataclass
class RandomShopper(LabelShopper):
    """
    This label shopper produces a random examples as their wishlist.
    """

    def wishlist(self, max_budget: float) -> List[Tuple[str, str]]:
        assert max_budget > 0
        cands = self.train_df[self.train_df['bought'] == False]
        cands = [(row.id, row.source_tok, row.cost) for _, row in cands.iterrows()]
        random.shuffle(cands)
        selection = []
        cur_budget = 0
        for _id, src_tok, cost in cands:
            if cur_budget + cost <= max_budget:
                selection.append((_id, src_tok))
                cur_budget += cost
        return selection


@dataclass
class DiverseShopper(LabelShopper):
    """This shopper produces diversified types as their wish list"""

    """minimum frequency for a type to consider as has enough labels"""
    min_freq: int = 5

    def wishlist(self, max_budget: float) -> List[Tuple[str, str]]:
        assert max_budget > 0

        owned = self.train_df[self.train_df['bought'] == True]
        covered_tfs = term_freqs(owned['source_tok'])

        # these have enough data or not needed in test set
        ignore_terms = set(t for t, f in covered_tfs.items() if f >= self.min_freq)
        test_tfs = term_freqs(self.test_df['source_tok'])
        buy_terms = {t: f for t, f in test_tfs.items() if t not in ignore_terms}

        cands = self.train_df[self.train_df['bought'] == False]
        cands = [(row.id, row.source_tok, row.cost) for _, row in cands.iterrows()]

        total = sum(buy_terms.values())
        term_inv_freq = {t: total / f for t, f in buy_terms.items()}
        ids = [i for i, _, c in cands]
        src_lines = [l for i, l, c in cands]
        values = np.array([sum(term_inv_freq.get(tok, 0) for tok in line.split())
                           for line in src_lines])
        costs = np.array([c for i, _, c in cands])
        worths = values / costs
        # on the API side, budget is calculated on the target side which we dont have access yet.
        # so I am going to try go over limit. API will stop me
        max_budget = max_budget * 1.05
        flags = self.pick_max_worth(costs, worths, max_budget=max_budget)
        assert len(flags) == len(ids)
        selection = [(i, l) for i, l, f in zip(ids, src_lines, flags) if f]
        return selection


class OverfitShopper(DiverseShopper):

    def wishlist(self, max_budget: float) -> List[Tuple[str, str]]:
        assert max_budget > 0
        owned = self.train_df[self.train_df['bought'] == True]
        covered_tfs = term_freqs(owned['source_tok'])
        cands = self.train_df[self.train_df['bought'] == False]
        # this is incomplete
        raise NotImplementedError()


label_shoppers = dict(
    random=RandomShopper,
    diverse=DiverseShopper,
    overfit=OverfitShopper
)
