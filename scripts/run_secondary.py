from pathlib import Path
import sys
import argparse
import json
import os
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--wait',action='store_true')
    a=p.parse_args()
    if a.wait:
        from provguard.core import ROOT,save_json
        while True:
            primary=ROOT/'reports/finalization_state.json'
            state=json.loads(primary.read_text()) if primary.exists() else {}
            if state.get('status')=='CORE_COMPLETE':
                break
            if state.get('status')=='FAILED':
                raise RuntimeError('Core finalization failed; secondary confirmation not started.')
            save_json(ROOT/'reports/secondary_controller_state.json',{'pid':os.getpid(),'status':'WAITING_CORE'})
            time.sleep(15)
    from provguard.secondary import run
    run()
