"""A temporary Windows execution request scoped to the live study process."""
import ctypes
import json
from pathlib import Path
import sys
import time
import psutil
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from provguard.core import ROOT,save_json

api=ctypes.windll.kernel32.SetThreadExecutionState
api.argtypes=[ctypes.c_uint32]
api.restype=ctypes.c_uint32
state=ROOT/'reports/awake_request.json'
try:
    if not api(0x80000001):
        raise OSError('Windows execution request failed.')
    save_json(state,{'status':'ACTIVE','scope':'Prevent automatic system sleep while the study process is live; request ends with this helper.'})
    while True:
        c=json.loads((ROOT/'reports/study_controller.json').read_text())
        path=ROOT/'reports/secondary_controller_state.json'
        s=json.loads(path.read_text()) if path.exists() else {}
        if s.get('status') in ['COMPLETE','PARTIAL'] or c.get('status')=='failed':
            break
        pid=s.get('pid') if s.get('status')=='RUNNING' else c.get('pid')
        if pid and not psutil.pid_exists(pid):
            break
        time.sleep(15)
finally:
    api(0x80000000)
    save_json(state,{'status':'RELEASED'})
