"""Optional real data fetch from original ETDataset public repository."""
import argparse, pathlib, urllib.request
from .data import load_etth1
URL='https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv'

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='data/ETTh1.csv');args=p.parse_args()
    path=pathlib.Path(args.output);path.parent.mkdir(exist_ok=True,parents=True)
    with urllib.request.urlopen(URL, timeout=60) as r: contents=r.read()
    path.write_bytes(contents)
    d=load_etth1(path)
    print('Loaded ETTh1 hourly transformer operating series:',len(d.values),'rows,',d.values.shape[1],'channels')
if __name__=='__main__':main()
