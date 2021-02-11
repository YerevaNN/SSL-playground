#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 2020-02-05

import argparse
import logging as log
import pprint
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import requests
from tqdm import tqdm

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

    @lru_cache()
    def get_task_metadata(self, task: str):
        headers = self.get_headers(user=True)
        end_pt = f"{self.base_url}/task_metadata/{task}"
        data = self.safe_get(end_pt, headers=headers)
        return data['task_metadata']

    @lru_cache()
    def get_data_metadata(self, dataset: str):
        headers = self.get_headers(user=True)
        end_pt = f"{self.base_url}/dataset_metadata/{dataset}"
        data = self.safe_get(end_pt, headers=headers)
        return data['dataset_metadata']

    def get_session_status(self):
        headers = self.get_headers(user=True, session=True)
        end_pt = f'{self.base_url}/session_status'
        data = self.safe_get(end_pt, headers=headers)
        return data['Session_Status']

    def query_labels(self, ids: List, batch_size=25_000):
        assert isinstance(ids, list)
        if isinstance(ids[0], int):
            ids = [str(i) for i in ids]
        res = {}
        status = None
        batches = [ids[j:j + batch_size] for j in range(0, len(ids), batch_size)]
        log.info(f"Going to ask {len(ids)} in {len(batches)} batches of size {batch_size}")
        for batch_ids in tqdm(batches):
            query = {'example_ids': batch_ids}
            headers = self.get_headers(user=True, session=True)
            # headers['Accept-Encoding'] = 'gzip, identity'  # we accept GZIP body
            # headers['Content-Encoding'] = 'gzip'  # we are sending GZIP body
            # data = zlib.compress(json.dumps(query).encode())
            r = requests.post(f"{self.base_url}/query_labels", json=query, headers=headers)
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
            status = data['Session_Status']
            labels = {ent['id']: ent['text'] for ent in data['Labels']}
            res.update(labels)
            if status['budget_left_until_checkpoint'] <= 0:
                break
        missing_count = sum(1 for id in ids if not res.get(id, None))
        log.warning(f'requested = {len(ids)}, missing: {missing_count}')
        return res, status

    def submit_predictions(self, test_ids, hyps):
        """This API for MT submissions"""
        assert isinstance(test_ids, list)
        assert isinstance(hyps, list)
        assert len(test_ids) == len(hyps)
        headers = self.get_headers(user=True, session=True)
        log.info("Making a submission")
        data = {'id': test_ids, 'text': hyps}
        data = {'predictions': data}
        # data = zlib.compress(json.dumps(data, ensure_ascii=False).encode())
        # headers['Content-Encoding'] = 'gzip'
        r = requests.post(f"{self.base_url}/submit_predictions", json=data, headers=headers)
        log.info(f'Status={r.status_code}')
        if r.status_code == 200:
            data = r.json()
            scores = data.get('Session_Status', {}).get('checkpoint_scores')
            if scores:
                log.info(pprint.pformat(scores, indent=2, width=80))
            return data
        else:
            try:
                data = r.json()
                trace = data.pop('trace', None)
                if trace:
                    log.warning(f"server trace: \n{trace}")
                log.warning(data)
            except:
                log.info(str(r.text))
                return False


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-sc', '--secret', required=True,
                   help='User Secret for LwLL API')
    p.add_argument('-ss', '--session',
                   help='Session Token to restore. When missing, it will create a new one')
    p.add_argument('-sn', '--session-name',
                   help='Session name when creating a new session. Ignored when session is restored')
    p.add_argument('-task', '--task', help='Task ID')
    p.add_argument('-dt', '--data_type', default='sample',
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


class LwZeroBudgetException(Exception):
    """
    Raise and catch this exception when the remaining budget is zero
    """
    pass
