from pathlib import Path
import argparse
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from provguard.robustness import prepare_reviews,run
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--prepare-reviews',action='store_true')
    a=p.parse_args()
    if a.prepare_reviews:
        print('Additional independent review candidates:',len(prepare_reviews()))
    else:
        run()
