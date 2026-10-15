"""Fresh-process final analysis after the sequential controller finishes."""
from pathlib import Path
import argparse
import json
import os
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from provguard.core import ROOT,sha,save_json
from provguard.study import PLAN

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--wait',action='store_true')
    args=parser.parse_args()
    state_path=ROOT/'reports/finalization_state.json'
    def state(status,**fields):
        save_json(state_path,{'pid':os.getpid(),'status':status,**fields})
    try:
        while True:
            registry=ROOT/'reports/experiment_registry.json'
            controller=ROOT/'reports/study_controller.json'
            r=json.loads(registry.read_text()) if registry.exists() else {}
            c=json.loads(controller.read_text()) if controller.exists() else {}
            done={(e['method'],e['seed']) for e in r.get('experiments',[]) if e['status']=='COMPLETED'}
            if all(key in done for key in PLAN) and c.get('status')!='running':
                break
            if c.get('status')=='failed':
                raise RuntimeError('Controller failed before all required runs completed: '+str(c.get('error')))
            if not args.wait:
                raise RuntimeError('Required core experiment matrix not complete.')
            state('WAITING_PRIMARY',completed_run_n=len(done),required_run_n=30)
            time.sleep(15)
        state('RUNNING_FINAL_ANALYSIS')
        from provguard.final_analysis import export_results
        export_results(final=True)
        from provguard.study import initialize
        save_json(ROOT/'reports/experiment_registry.json',initialize())
        save_json(ROOT/'reports/final_source_revision.json',{'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'src/provguard').glob('*.py'))},'script_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'scripts').glob('*.py'))},'protocol_sha256':sha(ROOT/'configs/protocol.json'),'results_master_sha256':sha(ROOT/'results_master.csv')})
        (ROOT/'reports/STATUS.md').write_text('# ProvGuard status\n\nAll 30 required DistilBERT runs are complete and verified. Final measured tables, statistics, and figures are in FINAL_EXPERIMENT_REPORT.md and figures/final/. Optional confirmations are reported separately when executed. Earlier paper files remain historical snapshots.\n',encoding='utf-8')
        state('CORE_COMPLETE',completed_run_n=30,report=str(ROOT/'reports/FINAL_EXPERIMENT_REPORT.md'))
    except Exception as exc:
        state('FAILED',error=repr(exc))
        raise

if __name__=='__main__':
    main()
