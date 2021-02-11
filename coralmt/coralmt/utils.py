#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 7/16/20

from pathlib import Path
from typing import Union, List, Iterator
from ruamel.yaml import YAML

import threading
import subprocess
from subprocess import Popen, PIPE
from multiprocessing.pool import ThreadPool
import collections as coll

import logging as log
log.basicConfig(level=log.ERROR)

yaml = YAML()

dir_path = Path(__file__).absolute().parent
script_path = dir_path / '../uroman/bin/uroman.pl'


def load_yaml(path: Union[str, Path]):

    with open(path, 'r') as fh:
        return yaml.load(fh)


def write_tsv(recs, path: Union[str, Path], delim='\t'):
    with open(path, 'w', encoding='utf-8', errors='ignore') as fh:
        for rec in recs:
            rec = delim.join(str(col) for col in rec)
            fh.write(rec)
            fh.write('\n')


def read_tsv(path: Union[str, Path], col=None, delim='\t'):
    with open(path, 'r', encoding='utf-8') as fh:
        for line in fh:
            rec = line.split(delim)
            if col is not None:
                rec = rec[col]
            yield rec


def uromanize_files(input_files: List[Path], output_files: List[Path], jobs: int = None,
                    language_code: str = 'ara'):
    """
    Language Code should be a dash followed by a 3 letter ISO code (lowercase)
        i.e. 'ara' for Arabic
    """
    assert len(input_files) == len(output_files)
    log.info(f" Going to uromanize {len(input_files)} files using pool of {jobs} jobs")

    def uromanize(inp_path: Path, out_path: Path):
        cmd = f'{script_path} -l {language_code} < {inp_path} > {out_path} '
        status, _ = subprocess.getstatusoutput(cmd)

    with ThreadPool(jobs) as pool:
        for (inp, out) in zip(input_files, output_files):
            pool.apply(uromanize, (inp, out))


def uromanize_lines(lines: Iterator[str]) -> Iterator[List[str]]:
    """Uromanize a stream of lines"""
    with Popen(str(script_path), stdin=PIPE, stdout=PIPE,
               universal_newlines=True, bufsize=1) as p:

        def write(iter, out):
            for i, line in enumerate(iter):
                if '\n' in line:
                    print(f'\n{i} :: {line} \n')
                    line = '.'
                out.write(line + '\n')
            out.close()

        writer = threading.Thread(target=write, args=(lines, p.stdin))
        # writer.daemon = True
        writer.start()

        for out_line in p.stdout:
            yield out_line.strip()
        log.debug(f'stopping {p.pid}')


def uromanize(line: str) -> List[str]:
    """Uromanize a single line .
    This is not efficient, please use uromanize_lines(...) instead
    """
    return [x for x in uromanize_lines([line])][0]


def term_freqs(lines):
    return coll.Counter(tok for line in lines for tok in line.split())
