#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu]
# Created: 6/25/20
import logging as log
import os
import sqlite3
import time
from datetime import timedelta
from itertools import zip_longest
from pathlib import Path
from typing import Iterator
from typing import List, Tuple, Union, Dict

import numpy as np
import pandas as pd
from rtg import log
from rtg.exp import TranslationExperiment as Experiment, TSVData, SqliteFile
from rtg.utils import IO
from sacremoses import MosesTokenizer, MosesDetokenizer


from .api import LwClient, LwZeroBudgetException
from .actlearn import label_shoppers
TextRec = Union[List[int], List[str]]
IdBiTextRec = Tuple[str, TextRec, TextRec]


class LwDataset:

    def __init__(self, data_dir: Path, lwc: LwClient):
        self.data_dir = data_dir
        log.info(f'Data dir = {data_dir}')
        assert self.data_dir.exists()

        self.lwc = lwc
        self.refresh_status()

    def refresh_status(self, status=None):
        self.status = status or self.lwc.get_session_status()
        self.active = self.status['active']
        self.phase = self.status['pair_stage']

        log.info(f"phase={self.phase}; active={self.active}")
        log.info(f'Remaining budget: {self.status["budget_left_until_checkpoint"]}')
        self.cur_data = self.status['current_dataset']
        log.info(f'dataset= {self.cur_data["name"]} type={self.cur_data["dataset_type"]}')
        self.is_sample = self.status['using_sample_datasets']
        suffix = "_sample" if self.is_sample else "_full"
        self.data_path = self.data_dir / self.cur_data['uid'] / (self.cur_data['name'] + suffix)
        log.info(f'Resolved data dir = {self.data_path}')
        assert self.data_path.exists()

    @property
    def is_complete(self):
        return self.active == 'Complete'


class LwDatasetMT(LwDataset):

    def __init__(self, data_dir: Path, lwc: LwClient, cache_dir: Path, shopper='random'):
        super().__init__(data_dir=data_dir, lwc=lwc)
        self.src, self.tgt = None, None
        self.src_tokr = MosesTokenizer()
        self.tgt_tokr = MosesTokenizer()
        self.tgt_detokr = MosesDetokenizer()
        self.cache_dir = cache_dir
        self._train_df = None
        self._test_df = None
        self._shopper = None
        assert shopper in label_shoppers
        self.shopper_type = label_shoppers[shopper]


    @property
    def dataset_name(self):
        return self.cur_data['uid']

    @property
    def train_df(self):
        if self._train_df is None:
            path = self.train_df_path
            if path.exists():
                df = pd.read_feather(path)
            else:
                # create a new one
                df = pd.read_feather(self.data_path / 'train_data.feather')
                df['source'] = df['source'].apply(lambda s: s or '...') # remove Nones
                df['cost'] = df['source'].apply(lambda s: len(s))
                df['source_tok'] = df['source'].apply(self.tokenize_src)
                df['bought'] = df['id'].apply(lambda s: False)
                df['target'] = df['id'].apply(lambda s: None)
                df['target_tok'] = df['id'].apply(lambda s: None)
                df.to_feather(path)
            self._train_df = df
        return self._train_df

    @property
    def test_df(self):
        if self._test_df is None:
            path = self.test_df_path
            if path.exists():
                df = pd.read_feather(path)
            else:
                df = pd.read_feather(self.data_path / 'test_data.feather')
                df['source'] = df['source'].apply(lambda s: s or '.')
                df['source_tok'] = df['source'].apply(self.tokenize_src)
                df['score'] =  df['id'].apply(lambda s: 0.0)
            self._test_df = df
        return self._test_df

    @property
    def test_df_path(self):
        path = self.cache_dir / (self.dataset_name + '-test.feather')
        return path

    @property
    def train_df_path(self):
        path = self.cache_dir / (self.dataset_name + '-train.feather')
        return path

    @property
    def label_shopper(self):
        if self._shopper is None:
            log.info(f"creating a new label shopper of type {self.shopper_type}")
            self._shopper = self.shopper_type(train_df=self.train_df, test_df=self.test_df)
        return self._shopper

    def invalidate_caches(self):
        self._train_df = None
        self._test_df = None
        self._shopper = None

    def refresh_status(self, status=None):
        old_data_path = getattr(self, 'data_path', None)
        super().refresh_status(status)
        assert self.cur_data["dataset_type"] == "machine_translation"
        if self.data_path != old_data_path:
            self.invalidate_caches()

    def submit_predictions(self, test_ids, test_hyps):
        res = self.lwc.submit_predictions(test_ids, test_hyps)
        if res and 'Session_Status' in res:
            self.refresh_status(status=res['Session_Status'])
        return res

    @property
    def remaining_budget(self):
        return self.status['budget_left_until_checkpoint']

    def tokenize_src(self, sent: str) -> str:
        return self.src_tokr.tokenize(sent, aggressive_dash_splits=True,
                                      return_str=True, escape=False)

    def tokenize_tgt(self, sent: str) -> str:
        return self.tgt_tokr.tokenize(sent, aggressive_dash_splits=True,
                                      return_str=True, escape=False)

    @classmethod
    def tokenize(cls, sents, tokr: MosesTokenizer):
        for sent in sents:
            sent_tok = tokr.tokenize(sent, aggressive_dash_splits=True,
                                     return_str=True, escape=False)
            yield sent_tok

    @classmethod
    def detokenize(cls, sents, detokr: MosesDetokenizer):
        for sent in sents:
            sent_detok = detokr.detokenize(sent.split(), return_str=True)
            yield sent_detok

    @staticmethod
    def tokenize_parallel_recs(ids: Iterator[str], srcs: Iterator[str], tgts: Iterator[str],
                               truncate: bool, src_len: int, tgt_len: int,
                               src_tokr, tgt_tokr) -> Iterator:
        recs = zip_longest(ids, srcs, tgts)
        recs = ((i, (x or "").strip(), (y or "").strip()) for i, x, y in recs)  # None -> Empty
        recs = ((i, x, y) for i, x, y in recs if x and y)  # skip empty lines
        recs = ((i, src_tokr(x), tgt_tokr(y)) for i, x, y in recs)

        if truncate:
            recs = ((i, src[:src_len], tgt[:tgt_len]) for i, src, tgt in recs)
        else:  # Filter out longer sentences
            recs = ((i, src, tgt) for i, src, tgt in recs if
                    len(src) <= src_len and len(tgt) <= tgt_len)
        return recs

    @staticmethod
    def write_parallel_recs(records: Iterator[IdBiTextRec], path: Union[str, Path]):
        seqs = ((i, ' '.join(map(str, x)), ' '.join(map(str, y))) for i, x, y in records)
        lines = (f'{i}\t{x}\t{y}' for i, x, y in seqs)
        TSVData.write_lines(lines, path)

    def prep_train_file(self, out_path: Path, chkpt: int, exp: Experiment):
        # get raw data : src, tgt
        log.info(f"Preparing training data for checkpoint {chkpt}")
        from coralmt import LwExperiment
        assert isinstance(exp, LwExperiment), f'{type(exp)} but expected {LwExperiment}'
        exp: LwExperiment = exp
        log.info(f"Going to query new data for checkpoint {chkpt}")
        ids, srcs, tgts = self.get_next_checkpoint_data()
        assert len(srcs) == len(tgts) == len(ids)
        assert len(ids) > 0
        log.info(f"Got new {len(ids)} sentences for training")

        # BPE and save it in sqlite DB
        args = exp.config['prep']
        s_time = time.time()

        parallel_recs = self.tokenize_parallel_recs(
            ids=ids, srcs=srcs, tgts=tgts, truncate=args['truncate'],
            src_len=args['src_len'], tgt_len=args['tgt_len'],
            src_tokr=exp.src_vocab.encode_as_ids, tgt_tokr=exp.tgt_vocab.encode_as_ids)

        if out_path.name.endswith('.db'):
            LwSqliteFile.write(out_path, records=parallel_recs, chkpt=chkpt)
        else:
            self.write_parallel_recs(parallel_recs, out_path)
        e_time = time.time()
        log.info(f"Time taken to process: {timedelta(seconds=(e_time - s_time))}")

    def get_next_checkpoint_data(self):
        """
        :return: id, src_tok, tgt_tok
        """
        self.refresh_status()
        if self.remaining_budget <= 0:  # empty
            log.warning("Remaining budget is zero")
            raise LwZeroBudgetException()
        wishlist = self.label_shopper.wishlist(max_budget=self.remaining_budget)
        ids = [i for i, s in wishlist]
        labels, status = self.lwc.query_labels(ids)
        res = [], [], []  # id, src, tgt
        lookup = {i:s for i, s in wishlist}
        for _id, ref in labels.items():
            ref_tok = self.tokenize_tgt(ref)
            src = lookup[_id]
            res[0].append(_id)
            res[1].append(src)
            res[2].append(ref_tok)
        self.label_shopper.commit(labels)
        self.train_df.to_feather(self.train_df_path)  # update

        self.refresh_status(status=status)
        return res

    def get_test_data(self, tokenized=False):
        test_ids = self.test_df['id'].tolist()
        if tokenized:
            test_srcs = self.test_df['source_tok'].tolist()
        else:
            test_srcs = self.test_df['source'].tolist()
        return test_ids, test_srcs


class LwSqliteFile(SqliteFile):
    APP_VERSION = 2  # for lwll
    # ID is set by LwLL System
    TABLE_STATEMENT = f"""CREATE TABLE IF NOT EXISTS data (
        id TEXT PRIMARY KEY NOT NULL,
        x BLOB NOT NULL,
        y BLOB,
        x_len INTEGER,
        y_len INTEGER,
        chkpt INT);"""
    INSERT_STMT = "INSERT OR REPLACE INTO data (id, x, y, x_len, y_len, chkpt) VALUES (?, ?, ?, ?, ?, ?)"

    @classmethod
    def write(cls, path, records: Iterator[IdBiTextRec], chkpt: int, remove_old=False):
        if path.exists() and remove_old:
            log.warning(f"Overwriting {path} with new records")
            os.remove(str(path))
        maybe_tmp = IO.maybe_tmpfs(path)
        log.info(f'Creating {maybe_tmp}')
        exists = maybe_tmp.exists()
        conn = sqlite3.connect(str(maybe_tmp))
        cur = conn.cursor()
        if not exists:
            # fresh DB; create tables and such
            cur.execute(cls.TABLE_STATEMENT)
            cur.execute(cls.INDEX_X_LEN)
            cur.execute(cls.INDEX_Y_LEN)
            cur.execute(f"PRAGMA user_version = {cls.CUR_VERSION};")
            cur.execute(f"PRAGMA application_id = {cls.APP_VERSION};")

        count = 0
        for id, x_seq, y_seq in records:
            # use numpy. its a lot efficient
            if not isinstance(x_seq, np.ndarray):
                x_seq = np.array(x_seq, dtype=np.int32)
            if y_seq is not None and not isinstance(y_seq, np.ndarray):
                y_seq = np.array(y_seq, dtype=np.int32)
            values = (id,
                      x_seq.tobytes(),
                      None if y_seq is None else y_seq.tobytes(),
                      len(x_seq),
                      len(y_seq) if y_seq is not None else -1,
                      chkpt)
            cur.execute(cls.INSERT_STMT, values)
            count += 1
        cur.close()
        conn.commit()
        if maybe_tmp != path:
            # bring the file back to original location where it should be
            IO.copy_file(maybe_tmp, path)
        log.info(f"stored {count} rows in {path}")
