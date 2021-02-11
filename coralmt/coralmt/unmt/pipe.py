#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 7/16/20
from pathlib import Path
from subprocess import CalledProcessError

from coralmt import LwZeroBudgetException, LwClient, LwDatasetMT, utils
import logging as log
from unmass.data.dictionary import Dictionary
from functools import partial
import subprocess

log.basicConfig(level=log.INFO)

uopen = partial(open, encoding='utf-8', errors='ignore')


class Pipeline:

    def __init__(self, root: Path):
        if not isinstance(root, Path):
            root = Path(root)
        self.root = root
        self._conf_file = root / 'conf.yml'
        assert self._conf_file.exists()
        self.conf = utils.load_yaml(self._conf_file)
        self.src, self.tgt = self.conf['langs']['src'], self.conf['langs']['tgt']

        # for unsup nmt, there is no source and target distinction by default;
        self.lang1, self.lang2 = list(sorted([self.src, self.tgt]))
        self.data_dir = self.root / 'data'
        full_vocab = self.data_dir / f"vocab.{self.lang1}-{self.lang2}"
        assert full_vocab.exists(), f'NOT FOUND: {full_vocab}'
        self.dico = Dictionary.read_vocab(full_vocab)
        assert 'lwll' in self.conf  # this is lwll experiment
        self.lw_conf = lw_conf = self.conf['lwll']
        self.lw_client = LwClient(base_url=lw_conf['base_url'],
                                  user_secret=lw_conf['user_secret'],
                                  session_token=lw_conf.get('session_token'))

        # assert (self.root / 'experiment' / 'checkpoint.pth').exists()

        self.task_id = lw_conf['task_id']
        if not self.lw_client.session_token:  # session token not initialized
            session_token = self.lw_client.new_session(
                task_id=self.task_id, data_type=lw_conf.get('data_type', 'full'),
                session_name=lw_conf.get('session_name', 'Coral uNMT'))
            log.info("Updating the config with session token")
            lw_conf['session_token'] = session_token
            # update conf on disk with session token
            self.store_config()

        self.dataset = LwDatasetMT(Path(lw_conf['data_path']), self.lw_client,
                                   cache_dir=self.data_dir)
        self.lw_chkpt = 0
        for p in self.data_dir.glob('selection-*.tsv'):
            pt = int(p.name.split('.')[0].split('-')[-1])
            self.lw_chkpt = max(self.lw_chkpt, pt)
        if self.lw_chkpt > 0:
            log.info(f"Last checkpoint = {self.lw_chkpt}")
        self._task_meta = None
        self._task_meta = None

    @property
    def task_meta(self):
        if not self._task_meta:
            self._task_meta = self.lw_client.get_task_metadata(self.task_id)
        return self._task_meta

    def store_config(self):
        utils.yaml.dump(self.conf, stream=self._conf_file)

    def pre_checks(self):
        log.warning("No pre checks yet")
        assert 'lwll' in self.conf
        assert 'unmt' in self.conf
        assert 'train' in self.conf['unmt']
        assert 'data_path' in self.conf['lwll']
        assert Path(self.conf['lwll']['data_path']).is_dir()

    def translate(self, ids, tok_srcs):
        assert len(ids) == len(tok_srcs)
        cmd = self.get_translate_cmd()
        res_dir = self.root / 'results'
        res_dir.mkdir(parents=True, exist_ok=True)
        res_file = res_dir / f'results.{self.lw_chkpt:02d}.out'
        cmd += ['--output', str(res_file)]
        log.info(f"RUN: {' '.join(cmd)}")
        lines = '\n'.join(tok_srcs)
        proc = subprocess.run(cmd, text=True, input=lines)
        if proc.returncode != 0:
            raise CalledProcessError(proc.returncode, cmd)
        log.info("Translation complete... ")
        assert res_file.exists()
        out_lines = res_file.read_text(encoding='utf-8', errors='ignore').splitlines(keepends=False)
        assert len(out_lines) == len(tok_srcs), \
            f'Sent source lines:{len(tok_srcs)}, but received:{len(out_lines)} translated lines'

        out_lines = [line.replace("@@ ", "") for line in out_lines]  # BPE undo
        return ids, out_lines

    def make_submission(self):
        test_ids, test_srcs = self.dataset.get_test_data()
        test_srcs = (src or '.' for src in test_srcs)  # remove None
        test_srcs_tok = list(self.dataset.tokenize(test_srcs, self.dataset.tgt_tokr))
        _, test_hyps = self.translate(test_ids, test_srcs_tok)
        test_hyps_detok = list(self.dataset.detokenize(test_hyps, self.dataset.tgt_detokr))
        return self.dataset.submit_predictions(test_ids, test_hyps_detok)

    def prepare_data(self):
        ids, src, ref = self.dataset.get_next_checkpoint_data()
        while True:
            path = self.data_dir / f'selection-{self.lw_chkpt:02d}.tsv'
            if not path.exists():
                break
            self.lw_chkpt += 1
        utils.write_tsv(recs=zip(ids, src, ref), path=path)
        # merge
        all_recs = [path]  # current file
        for pt in range(1, self.lw_chkpt):  # older files
            path = self.data_dir / f'selection-{pt:02d}.tsv'
            if not path.exists():
                log.warning(f"NOT found {path}; this is an error")
                continue
            all_recs.append(path)

        train_src = self.data_dir / 'train.raw.src.tok'
        train_tgt = self.data_dir / 'train.raw.tgt.tok'
        self.maybe_backup(train_src, train_tgt)

        tokr_args = dict(aggressive_dash_splits=True, return_str=True, escape=False)
        src_tokr = partial(self.dataset.src_tokr.tokenize, **tokr_args)
        tgt_tokr = partial(self.dataset.tgt_tokr.tokenize, **tokr_args)
        with uopen(train_src, 'w') as src_f, uopen(train_tgt, 'w') as tgt_f:
            for path in all_recs:
                for rec in utils.read_tsv(path):
                    if len(rec) != 3:
                        log.warning(f"Skip record: {rec}")
                        continue
                    _id, src, tgt = rec
                    src, tgt = src_tokr(src.strip()), tgt_tokr(tgt.strip())
                    if not src or not tgt:
                        log.info(f'skip empty record {id}')
                    src_f.write(f'{src}\n')
                    tgt_f.write(f'{tgt}\n')
        train_src_bin = self.data_dir / f'train.{self.src}-{self.tgt}.{self.src}.pth'
        train_tgt_bin = self.data_dir / f'train.{self.src}-{self.tgt}.{self.tgt}.pth'
        self.maybe_backup(train_src_bin, train_tgt_bin)
        log.info(f"{train_src} --> {train_src_bin}")
        Dictionary.index_data(str(train_src), str(train_src_bin), self.dico)
        log.info(f"{train_tgt} --> {train_tgt_bin}")
        Dictionary.index_data(str(train_tgt), str(train_tgt_bin), self.dico)

    def maybe_backup(self, *paths: Path):
        for path in paths:
            if path.exists():
                bak = path.with_suffix(path.suffix + f'.{self.lw_chkpt - 1:02d}')
                log.info(f"{path} -> {bak}")
                path.rename(bak)

    def get_trainer_cmd(self):
        l1, l2 = self.lang1, self.lang2
        args = self.conf['unmt']['train']
        assert 'data_path' not in args
        assert 'exp_path' not in args
        args['lgs'] = args.get('lgs', f'{l1}-{l2}')
        args['mt_steps'] = args.get('mt_steps', f'{l1}-{l2},{l2}-{l1}')
        cmd = [
            'python', '-m', 'unmass.train',
            '--data_path', str(self.data_dir.absolute()),
            '--exp_path', str((self.root / 'experiment').absolute())
        ]

        for name, val in args.items():
            cmd += [f'--{name}', f'{val}']
        return cmd

    def get_translate_cmd(self):
        args = self.conf['unmt']['translate']
        for name, default in [['src_lang', self.src],
                              ['tgt_lang', self.tgt],
                              ['length_penalty', 0.6],
                              ['beam', 4],
                              ['batch', 64]]:
            args[name] = args.get(name, default)
        assert 'model' not in args
        cmd = [
            'python', '-m', 'unmass.translate',
            '--model', str((self.root / 'experiment/checkpoint.pth').absolute())]

        for name, val in args.items():
            cmd += [f'--{name}', f'{val}']

        return cmd

    def train(self):
        self.dataset.refresh_status()
        self.prepare_data()
        cmd = self.get_trainer_cmd()
        log.info(f"RUN: {' '.join(cmd)}")
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            raise CalledProcessError(proc.returncode, cmd)
        log.info("Training complete... ")

    def run(self):
        # prepare
        self.pre_checks()
        aborted = False
        while not self.dataset.is_complete:
            self.lw_chkpt += 1
            log.info(f"Check Point = {self.lw_chkpt}")
            try:
                self.train()
            except LwZeroBudgetException as _:
                self.lw_chkpt -= 1
                log.warning("Budget exhausted... Training is either stopped or failed.")
                # go make a submission, next
            if not self.make_submission():
                log.warning("Aborting... Looks like there was an error while submitting")
                aborted = True
                break

        if not aborted:
            log.info("Success")


def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('exp', help='experiment path')
    return vars(p.parse_args())


def main():
    args = parse_args()
    pipe = Pipeline(root=args.pop('exp'))
    pipe.run()


if __name__ == '__main__':
    main()
