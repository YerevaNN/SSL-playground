#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 7/28/20
import logging as log
from pathlib import Path
import pandas as pd

log.basicConfig(level=log.INFO)

def main(args=None):
    args = args or parse_args()
    out = args.out
    dff: Path = args.inp
    assert dff.exists()
    df = pd.read_feather(str(dff))
    cols = list(df.columns)
    count = 0
    for idx, row in df.iterrows():
        line = "\t".join(str(row[col]).replace("\t", " ") for col in cols)
        line = line.replace("\n", " ")
        out.write(f"{line}\n")
        count += 1
    log.info(f"wrote {count} lines to {out}")

def parse_args():
    import argparse
    import sys
    import io
    stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('-i', '--inp', type=Path, required=True, help='Dataframe feather file')

    p.add_argument('-o', '--out', type=argparse.FileType('w'), default=stdout,
                   help='Output file path')
    return p.parse_args()


if __name__ == '__main__':
    main()
