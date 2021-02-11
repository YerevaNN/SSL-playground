#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 7/1/20
import copy
import time
from datetime import timedelta
from pathlib import Path

import pandas as pd
from rtg import TranslationExperiment, log, yaml, BatchIterable
from rtg.data.dataset import LoopingIterable, TSVData
from rtg.utils import IO
from rtg.module.decoder import Decoder
import io
import torch

from coralmt import LwClient, LwDatasetMT, LwZeroBudgetException
from coralmt.utils import uromanize_files


class LwExperiment(TranslationExperiment):

    def __init__(self, work_dir, read_only=False, config=None):
        super().__init__(work_dir=work_dir, read_only=read_only, config=config)
        self.lwll_dir = self.work_dir / 'lwll'
        if not self.lwll_dir.is_dir():
            self.lwll_dir.mkdir(parents=True)
        assert 'lwll' in self.config  # this is lwll experiment
        self.lw_conf = lw_conf = self.config['lwll']
        self.lw_client = LwClient(base_url=lw_conf['base_url'],
                                  user_secret=lw_conf['user_secret'],
                                  session_token=lw_conf.get('session_token'))

        self.task_id = lw_conf['task_id']

        update_conf = False
        if not self.read_only:
            if 'shopper' not in self.lw_conf:
                self.lw_conf['shopper'] = self.lw_conf.get('shopper', dict(type='random'))
                update_conf = True
            if not self.lw_client.session_token:  # session token not initialized
                session_token = self.lw_client.new_session(
                    task_id=self.task_id, data_type=lw_conf.get('data_type', 'full'),
                    session_name=lw_conf.get('session_name', 'RTG NMT auto'))
                log.info("Updating the config with session token")
                lw_conf['session_token'] = session_token
                # update conf on disk with session token
                update_conf = True

        if update_conf:
            self.store_config()

        shopper = self.lw_conf.get('shopper', {}).get('type', 'random')
        self.dataset = LwDatasetMT(Path(lw_conf['data_path']), self.lw_client,
                                   cache_dir=self.lwll_dir, shopper=shopper)
        self._task_meta = None
        self.lw_chkpt = 0
        self._trainer = None
        self._decoder = None

    @property
    def trainer(self):
        if not self._trainer:  # lazy init
            from rtg.registry import trainers, factories
            name, optim_args = self.optim_args
            self._trainer = trainers[self.model_type](
                self, optim=name, model_factory=factories[self.model_type], **optim_args)
        return self._trainer

    @property
    def decoder(self):
        if not self._decoder:
            # FIXME:  support for ensembling last n checkpoints on disk
            self._decoder = Decoder.new(self, self.trainer.model)
        return self._decoder

    @property
    def task_meta(self):
        if not self._task_meta:
            self._task_meta = self.lw_client.get_task_metadata(self.task_id)
        return self._task_meta

    def pre_process(self, args=None, force=False):
        if self.has_prepared() and not force:
            log.warning("Already prepared")
            return
        args = args if args else self.config['prep']
        if 'parent' in self.config:
            self.inherit_parent()

        # base_dataset = self.lw_client.get_task_metadata(self.task_meta['base_dataset'])
        # adapt_dataset = self.lw_client.get_task_metadata(self.task_meta['adaptation_dataset'])
        s_time = time.time()
        base_dataset = self.task_meta['base_dataset']
        adapt_dataset = self.task_meta['adaptation_dataset']
        lwll_data = Path(self.lw_conf['data_path'])
        suffix = '_full'
        all_srcs, para_srcs = [], []
        for data_name in [base_dataset, adapt_dataset]:
            train_df = lwll_data / data_name / (data_name + suffix) / 'train_data.feather'
            train_src_txt = self.lwll_dir / (data_name + '.train.src')
            self._extract_df_col_as_text(train_df, col_name='source', out=train_src_txt)
            all_srcs.append(train_src_txt), para_srcs.append(train_src_txt)

            test = lwll_data / data_name / (data_name + suffix) / 'test_data.feather'
            train_src_txt = self.lwll_dir / (data_name + '.test.src')
            self._extract_df_col_as_text(test, col_name='source', out=train_src_txt)
            all_srcs.append(train_src_txt)

        try:
            if self.config['prep']['uromanize']:
                urom_srcs = [src.with_suffix('.rom.src') for src in all_srcs]
                uromanize_files(all_srcs, urom_srcs, jobs=len(all_srcs), language_code='-ara')
                para_srcs = [src.with_suffix('.rom.src') for src in para_srcs]
        except KeyError:
            pass   # Backwards compatibility

        args['mono_src'] = list(set(args.get('mono_src', []) + [str(x) for x in para_srcs]))
        self.make_vocabs(args)

        e_time = time.time()
        log.info(f"Time taken to process: {timedelta(seconds=(e_time - s_time))}")

        # Note: the data is not encoded using vocab
        self._prepared_flag.touch()
        self.persist_state()

    def make_vocabs(self, args):

        xt_args = dict(no_split_toks=args.get('no_split_toks'),
                       char_coverage=args.get('char_coverage', 0))
        if args.get('shared_vocab'):  # shared vocab
            corpus = [args[key] for key in ['train_src', 'train_tgt', 'mono_src', 'mono_tgt']
                      if args.get(key)]
            self.shared_field = self._make_vocab("shared", self._shared_field_file, args['pieces'],
                                                 args['max_types'], corpus=corpus, **xt_args)
        else:  # separate vocabularies
            src_corpus = [args[key] for key in ['train_src', 'mono_src'] if args.get(key)]
            self.src_field = self._make_vocab("src", self._src_field_file, args['pieces'],
                                              args['max_src_types'], corpus=src_corpus, **xt_args)

            # target vocabulary
            tgt_corpus = [args[key] for key in ['train_tgt', 'mono_tgt'] if args.get(key)]
            self.tgt_field = self._make_vocab("src", self._tgt_field_file, args['pieces'],
                                              args['max_tgt_types'], corpus=tgt_corpus, **xt_args)

    @classmethod
    def _extract_df_col_as_text(cls, feather, col_name, out: Path):
        log.info(f"Column={col_name} {feather} --> {out}")
        df = pd.read_feather(feather)
        count = 0
        with IO.writer(out) as wrt:
            for idx, row in df.iterrows():
                line = (row[col_name] or "").strip()
                wrt.write(f'{line}\n')
                count += 1
        log.info(f"Wrote {count} lines to {out}")
        return count

    def train(self, args=None):
        run_args = copy.deepcopy(self.config.get('trainer', {}))
        if args:
            run_args.update(args)
        if 'init_args' in run_args:
            del run_args['init_args']
        train_steps = run_args['steps']

        _, last_step = self.get_last_saved_model()
        if self._trained_flag.exists():
            # noinspection PyBroadException
            try:
                last_step = max(last_step, yaml.load(self._trained_flag.read_text())['steps'])
            except Exception as _:
                pass

        if last_step < train_steps:  # regular training
            stopped = self.trainer.train(fine_tune=False, **run_args)
            final_step = self.trainer.opt.curr_step
            if not self.read_only:
                status = dict(steps=final_step, early_stopped=stopped, finetune=False)
                try:
                    status['earlier'] = yaml.load(self._trained_flag.read_text())
                except Exception as _:
                    pass
                yaml.dump(status, stream=self._trained_flag)
            return final_step
        else:
            return last_step

    def get_train_data(self, batch_size: int, steps: int = 0, sort_by='eq_len_rand_batch',
                       batch_first=True, shuffle=False, fine_tune=False, keep_in_mem=False):
        self.lw_chkpt += 1
        train_db = self.train_db
        try:
            self.dataset.prep_train_file(out_path=train_db, chkpt=self.lw_chkpt, exp=self)
        except LwZeroBudgetException as e:
            if not train_db.exists():
                raise e
            # there is already some training data. just retrain and loop
            log.warning("Remaining budget is zero. reusing old training data")
        train_db = IO.maybe_tmpfs(train_db)
        train_data = BatchIterable(train_db, batch_size=batch_size, sort_by=sort_by,
                                   batch_first=batch_first, shuffle=shuffle, field=self.tgt_vocab,
                                   keep_in_mem=keep_in_mem, **self._get_batch_args())
        if steps > 0:
            train_data = LoopingIterable(train_data, steps)
        return train_data

    def get_val_data(self, *args, **kwargs):
        if not self.valid_file.exists():  # hack : copy some training data as valid; TODO: exclude
            val_size = 50
            log.warning(
                f"FIXME: Using subset of training data (batches={val_size}) for validation.")

            def head():
                train_data = BatchIterable(self.train_db, batch_size=1024, sort_by=None,
                                           batch_first=True, shuffle=False, field=self.tgt_vocab,
                                           keep_in_mem=False, **self._get_batch_args())
                count = 0
                for batch in train_data:
                    for i in range(len(batch)):
                        x, y = batch.x_seqs[i], batch.y_seqs[i]
                        x = x[:batch.x_len[i]].tolist()
                        y = y[:batch.y_len[i]].tolist()
                        yield x, y
                        count += 1
                        if count >= val_size:
                            break

            log.info(f"Writing recs to {self.valid_file}")
            TSVData.write_parallel_recs(head(), self.valid_file)

        # delegate the rest of work to parent
        return super(LwExperiment, self).get_val_data(*args, **kwargs)

    def translate(self, test_ids, test_srcs):
        assert len(test_ids) == len(test_srcs)
        with torch.no_grad():
            self.decoder.model.eval()
            out_buffer = io.StringIO()
            dec_args = self.config.get('decoder') or self.config['tester']['decoder']
            dec_args = copy.deepcopy(dec_args)
            dec_args['batch_size'] = dec_args['batch_size'] / dec_args['beam_size']
            self.decoder.decode_file(test_srcs, out=out_buffer, **dec_args)
            out_buffer.flush()

            hyps = []
            for line in out_buffer.getvalue().splitlines(keepends=True):
                hyp, _ = line.rstrip('\n').split('\t')
                hyp = self.dataset.tgt_detokr.detokenize(tokens=hyp.split())
                hyps.append(hyp)
            assert len(test_ids) == len(hyps), f'ids == hyps ?= {len(test_ids)} == {len(hyps)}'
            self.decoder.model.train()
            return test_ids, hyps

