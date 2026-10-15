"""Read live experiment progress without importing GPU libraries."""
import json
from pathlib import Path
root = Path(__file__).resolve().parents[1]
for name in ["study_controller"]:
    path = root / "reports" / f"{name}.json"
    if path.exists():
        print(f"{name}: {path.read_text(encoding='utf-8')}")
registry=root/'reports/experiment_registry.json'
if registry.exists():
    entries=json.loads(registry.read_text())['experiments']
    print('Completion matrix (required 30 runs):')
    for method in dict.fromkeys(e['method'] for e in entries):
        print(method+': '+', '.join(str(e['seed'])+'='+e['status'] for e in entries if e['method']==method))
for status in sorted((root / "outputs/provguard").glob("*/seed_*/status.json")):
    state = json.loads(status.read_text())
    print(f"{status.parent.parent.name}/{status.parent.name}: {state['status']}")
