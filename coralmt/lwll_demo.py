#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 2020-02-05

import argparse
import logging as log
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests

log.basicConfig(level=log.INFO)


class LwClient:

    def __init__(self, base_url: str, user_secret: str,
                 session_token: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.user_secret = user_secret
        self.session_token = session_token
        log.info(f'Created Client with bas URL: {self.base_url}')
        if not self.session_token:
            log.info(f'session_token is not set, call new_session(...)')
        else:
            log.info(f'Session {session_token} restored')

    def get_headers(self, user=False, session=False):
        h = {}
        if user:
            h['user_secret'] = self.user_secret
        if session:
            assert self.session_token, 'session_token not set. call new_session()'
            h['session_token'] = self.session_token
        return h

    def new_session(self, task_id: str, data_type: str, overwrite: bool = False,
                    session_name: str = 'unknown'):
        assert data_type in ('full', 'sample')
        if not overwrite and self.session_token:
            raise Exception('Please pass overwrite=False to overwrite a session token')
        log.info(f"Creating a new session for task {task_id}; data_type={data_type}")
        headers = self.get_headers(user=True)
        end_pt = f"{self.base_url}/auth/create_session"
        data = dict(session_name=session_name, data_type=data_type, task_id=task_id)
        data = requests.post(end_pt, headers=headers, json=data).json()
        self.session_token = data['session_token']
        log.info(f'Session; {session_name} = {self.session_token}')
        with Path('~/.lwsessions').expanduser().open('a') as wrt:
            wrt.write(f'{datetime.now()} {session_name} {self.session_token}')
        return self.session_token

    def safe_get(self, url: str, safe_codes=(200,), **kwargs):
        r = requests.get(url, **kwargs)
        if r.status_code not in safe_codes:
            log.warning(r.content)
            raise Exception(r)
        return r.json()

    def list_tasks(self):
        """Gets a list of active task names"""
        headers = self.get_headers(user=True)
        data = self.safe_get(f"{self.base_url}/list_tasks", headers=headers)
        return data['tasks']

    def get_task_metadata(self, task: str):
        headers = self.get_headers(user=True)
        end_pt = f"{self.base_url}/task_metadata/{task}"
        data = self.safe_get(end_pt, headers=headers)
        return data['task_metadata']

    def get_session_status(self):
        headers = self.get_headers(user=True, session=True)
        end_pt = f'{self.base_url}/session_status'
        data = self.safe_get(end_pt, headers=headers)
        return data['Session_Status']

    def query_labels(self, ids: List):
        assert isinstance(ids, list)
        if isinstance(ids[0], int):
            ids = [str(i) for i in ids]
        query = {'example_ids': ids}
        headers = self.get_headers(user=True, session=True)
        r = requests.post(f"{self.base_url}/query_labels", json=query, headers=headers)
        log.info(f"Status = {r.status_code}")
        if r.status_code != 200:
            # noinspection PyBroadException
            try:
                data = r.json()
                if 'trace' in data:
                    log.warning("SERVER TRACE:\n" + data['trace'])
                    del data['trace']
            except Exception as _:
                data = r.text
            raise Exception(f'{r.status_code}\n {data}')
        data = r.json()
        labels = data['Labels']
        status = data['Session_Status']

        labels = {ent['id']: ent['text'] for ent in labels}
        refs = [labels.get(i, None) for i in ids]
        missing_count = sum(1 for t in refs if not t)
        log.warning(f'requested = {len(ids)}, missing: {missing_count}')
        # TODO: handle missing recs properly
        return refs, status

    def submit_predictions(self, test_ids, hyps):
        """This API for MT submissions"""
        assert isinstance(test_ids, list)
        assert isinstance(hyps, list)
        assert len(test_ids) == len(hyps)
        headers = self.get_headers(user=True, session=True)
        log.info("Making a submission")
        data = {'id': test_ids, 'text': hyps}
        data = {'predictions': data}
        r = requests.post(f"{self.base_url}/submit_predictions", json=data, headers=headers)
        log.info(f'Status={r.status_code}')
        log.info(str(r.text))
        # TODO: parse result and return instead of log
        return r.status_code == 200


class LwDataset:

    def __init__(self, data_dir: Path, lwc: LwClient):
        self.data_dir = data_dir
        log.info(f'Data dir = {data_dir}')
        assert self.data_dir
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

    def remaining_budget(self):
        pass


class LwDatasetMT(LwDataset):

    def refresh_status(self, status=None):
        super().refresh_status(status)
        assert self.cur_data["dataset_type"] == "machine_translation"
        log.info(f"Lang Pairs: {self.cur_data['language_from']} -> {self.cur_data['language_to']}")
        self.train_data = pd.read_feather(self.data_path / 'train_data.feather')
        self.test_data = pd.read_feather(self.data_path / 'test_data.feather')

    def get_next_checkpoint_data(self):
        rem_budget = self.status['budget_left_until_checkpoint']
        rem_budget = min(rem_budget, len(self.train_data))
        if rem_budget <= 0:
            # empty
            return [], []
        cp_train = self.train_data.sample(rem_budget)
        ids = cp_train['id'].tolist()
        srcs = cp_train['source'].tolist()
        tgts, status = self.lwc.query_labels(ids)
        self.refresh_status(status)
        return srcs, tgts

    def get_test_data(self):
        test_ids = self.test_data['id'].tolist()
        test_srcs = self.test_data['source'].tolist()
        return test_ids, test_srcs


class DummyModel:
    def __init__(self, mode='copy'):
        self.mode = mode

    def train(self, test_srcs, test_tgts):
        # No-Op
        log.warning("train() is called. It is dummy")
        pass

    def translate(self, test_ids, test_srcs):
        log.warning("translate() is called. It is dummy")
        if self.mode == 'copy':
            # dont send None; API crashes if you do
            test_hyps = [t or 'empty' for t in test_srcs]  # return source as translations; dummy
        elif self.mode == 'quick':
            quick = 'The quick brown fox jumps over the lazy dog'  # the one JPL used for testing
            test_hyps = [quick for t in test_srcs]  # return some
        elif self.mode == 'short':
            test_hyps = ['a' for t in test_srcs]  # smaller dummy outputs
        else:
            raise Exception(f"Unknown {self.mode}; Known= copy, quick, short")
        return test_ids, test_hyps


class Pipeline:

    def __init__(self, model, dataset: LwDatasetMT):
        self.model = model
        self.dataset = dataset

    def make_submission(self):
        test_ids, test_srcs = self.dataset.get_test_data()
        _, test_hyps = self.model.translate(test_ids, test_srcs)
        return self.dataset.lwc.submit_predictions(test_ids, test_hyps)

    def run(self):
        aborted = False
        count = 0
        while not self.dataset.is_complete:
            count += 1
            log.info(f"Check Point = {count}")
            self.dataset.refresh_status()
            train_srcs, train_tgts = self.dataset.get_next_checkpoint_data()
            self.model.train(train_srcs, train_tgts)
            if not self.make_submission():
                log.warning("Aborting... Looks like there was an error while submitting")
                aborted = True
                break
        if not aborted:
            log.info("Success")


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-sc', '--secret', required=True,
                   help='User Secret for LwLL API')
    p.add_argument('-ss', '--session',
                   help='Session Token to restore. When missing, it will create a new one')
    p.add_argument('-sn', '--session-name',
                   help='Session name when creating a new session. Ignored when session is restored')
    p.add_argument('-task', '--task', help='Task ID')
    p.add_argument('-dt', '--data_type', default='full',
                   choices=['full', 'sample'], help='Data Type')
    p.add_argument('-dd', '--data_dir', default=Path('/datasets'), type=Path,
                   help='data dir where lwll datasets are downloaded locally')
    p.add_argument('-dummy', '--dummy', default='copy', choices=['copy', 'short', 'quick'],
                   help='dummy mode; copy = copy source as ouput. short= short outputs;'
                        ' quick=The quick brown fox jumped over a lazy dog.')
    p.add_argument('-lt', '--list-tasks', action='store_true', help='List tasks')
    args = vars(p.parse_args())
    return args


def list_tasks(lwc: LwClient, problem_type='machine_translation'):
    tasks = lwc.list_tasks()
    metas = []
    for i, task in enumerate(tasks):
        md = lwc.get_task_metadata(task)
        if md['problem_type'] != problem_type:
            log.warning(f"Skip {task} of type {md['problem_type']}")
            continue
        metas.append(md)
    return metas


def main():
    args = parse_args()
    # EDIT THESE
    data_dir = args['data_dir']
    assert data_dir.exists()
    user_secret = args['secret']
    # url = 'http://myserviceloadbalancer-679310346.us-east-1.elb.amazonaws.com'
    url = "https://api-dev.lollllz.com/"

    lwc = LwClient(base_url=url, user_secret=user_secret)

    if args.get('list_tasks'):
        tasks = list_tasks(lwc)
        import json
        for task in tasks:
            print(json.dumps(task, indent=2, ensure_ascii=False))
        return
    if not args.get('task'):
        log.error("Please set --task=<task_id> . You may get <task_id> by using --list-tasks")
        return

    if args.get('session'):
        # Option 2: restore session
        lwc.session_token = args.get('session')
    else:
        # Option 1: new session
        lwc.new_session(task_id=args['task'], data_type=args['data_type'],
                        session_name=args['session_name'])

    log.info(lwc.get_session_status())

    dataset = LwDatasetMT(data_dir, lwc)
    model = DummyModel(mode=args['dummy'])
    pipe = Pipeline(model, dataset=dataset)
    pipe.run()
    # pipe.make_submission()


if __name__ == '__main__':
    main()
