#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 6/24/20

from pathlib import Path

import torch
from rtg import log
from rtg.pipeline import Pipeline as BasePipeline

from coralmt import LwExperiment, LwZeroBudgetException
from coralmt.utils import uromanize_lines


class Pipeline(BasePipeline):

    def __init__(self, exp: LwExperiment):
        assert isinstance(exp, LwExperiment)
        super().__init__(exp=exp)
        self.exp: LwExperiment = exp
        self.dataset = self.exp.dataset

    def make_submission(self):
        test_ids, test_srcs_tok = self.dataset.get_test_data(tokenized=True)
        test_srcs_tok = [src or '.' for src in test_srcs_tok]   # remove None
        if self.exp.config['prep'].get('uromanize'):
            test_srcs_tok = list(uromanize_lines(test_srcs_tok))
        _, test_hyps = self.exp.translate(test_ids, test_srcs_tok)
        test_hyps_detok = list(self.dataset.detokenize(test_hyps, self.dataset.tgt_detokr))
        return self.dataset.submit_predictions(test_ids, test_hyps_detok)

    def pre_checks(self):
        # mono datasets exist
        mono_src = self.exp.config['prep'].get('mono_src')
        if mono_src:
            assert all(Path(x).exists() for x in mono_src)

        mono_tgt = self.exp.config['prep'].get('mono_tgt')
        if mono_tgt:
            assert all(Path(x).exists() for x in mono_tgt)

        base_dataset = self.exp.task_meta['base_dataset']
        adaptation_dataset = self.exp.task_meta['base_dataset']

        data_dir = Path(self.exp.config['lwll']['data_path'])
        assert data_dir.exists()

        suffix = '_full'
        for dataset in [base_dataset, adaptation_dataset]:
            data_path = data_dir / dataset / (dataset + suffix)
            assert data_path.exists()
            assert (data_path / 'train_data.feather').exists()
            assert (data_path / 'test_data.feather').exists()

    def run(self, run_test=False):
        # prepare
        self.pre_checks()
        self.exp.pre_process(force=False)
        aborted = False
        count = 0
        all_train_steps = self.exp.config['trainer']['steps']
        train_steps = all_train_steps[count] if isinstance(all_train_steps, list) else all_train_steps
        while not self.dataset.is_complete:
            count += 1
            log.info(f"Check Point = {count}")
            self.dataset.refresh_status()
            try:
                train_steps = self.exp.train(dict(steps=train_steps))
                if isinstance(all_train_steps, list):
                    count_idx = count if count < len(all_train_steps) else -1
                    train_steps += all_train_steps[count_idx]
                else:
                    train_steps += all_train_steps
            except LwZeroBudgetException as _:
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
    parser = argparse.ArgumentParser(prog="coral-snmt", description="Coral Supervised NMT")
    parser.add_argument("exp", help="Working directory of experiment", type=Path)
    parser.add_argument("conf", type=Path, nargs='?',
                        help="Config File. By default <work_dir>/conf.yml is used")
    parser.add_argument("-G", "--gpu-only", action="store_true", default=False,
                        help="Crash if no GPU is available")
    args = parser.parse_args()

    if args.gpu_only:
        assert torch.cuda.is_available(), "No GPU found... exiting"
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            log.info(f'Cuda {i}: {torch.cuda.get_device_properties(i)}')

    conf_file: Path = args.conf if args.conf else args.exp / 'conf.yml'
    assert conf_file.exists(), f'NOT FOUND: {conf_file}'
    return LwExperiment(args.exp, config=conf_file)


def main():
    pipe = Pipeline(exp=parse_args())
    pipe.run()


if __name__ == '__main__':
    main()
