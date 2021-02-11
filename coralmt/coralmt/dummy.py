#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 6/24/20

from pathlib import Path

from rtg import log
from rtg.pipeline import Pipeline as BasePipeline
import json

from coralmt import LwExperiment


class Pipeline(BasePipeline):

    def __init__(self, exp: LwExperiment, base_test: Path, adaptation_test: Path):
        assert isinstance(exp, LwExperiment)
        super().__init__(exp=exp)
        self.exp: LwExperiment = exp
        self.dataset = exp.dataset
        assert base_test.exists()
        assert adaptation_test.exists()
        self.base_set = base_test
        self.adaptation_set = adaptation_test
        self.meta_file = self.exp.work_dir / 'meta.json'
        self.meta = dict(checkpt=0)
        if self.meta_file.exists():
            self.load_meta()

        self.n_base_checkpts = len(self.exp.task_meta['base_label_budget_full'])
        #self.adapt_checkpts = len(self.exp.task_meta['adaptation_label_budget_full'])

    def store_meta(self):
        with self.meta_file.open('w') as f:
            json.dump(self.meta, f)

    def load_meta(self):
        with self.meta_file.open('r') as f:
            self.meta = json.load(f)

    def read_tsv(self, path: Path):
        with path.open(mode='r', encoding='utf-8') as lines:
            return [line.split('\t') for line in lines]

    def make_submission(self):
        file = self.base_set
        if self.meta['checkpt'] >= self.n_base_checkpts:
            file = self.adaptation_set

        recs = self.read_tsv(file)
        ids = [r[0] for r in recs]
        texts = [r[1] for r in recs]

        return self.dataset.submit_predictions(ids, texts)

    def run(self, run_test=False):
        aborted = False
        while not self.dataset.is_complete:
            log.info(f"{self.meta}")
            self.dataset.refresh_status()
            if self.make_submission():
                self.meta['checkpt'] += 1
                self.store_meta()
            else:
                log.warning("Aborting... Looks like there was an error while submitting")
                aborted = True
                break
        if not aborted:
            self.meta['success'] = True
            log.info("Success")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(prog="coralmt", description="Dummy pipeline")
    parser.add_argument("exp", help="Working directory of experiment", type=Path)
    parser.add_argument("--base", required=True,
                        help="Base test set <id><tab><hyp> to submit", type=Path)
    parser.add_argument("--adapt",required=True,
                        help="Adaptation test set <id><tab><hyp> to submit", type=Path)
    args = parser.parse_args()
    return vars(args)

def main():
    args = parse_args()
    exp = LwExperiment(args['exp'])
    pipe = Pipeline(exp=exp, base_test=args['base'], adaptation_test=args['adapt'])
    pipe.run()


if __name__ == '__main__':
    main()
