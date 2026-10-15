from pathlib import Path
import argparse
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from provguard.secondary import train_deberta
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--method',required=True,choices=['baseline','provguard_1000'])
    p.add_argument('--seed',required=True,type=int)
    p.add_argument('--augment-train')
    a=p.parse_args()
    train_deberta(a.method,a.seed,Path(a.augment_train) if a.augment_train else None)

