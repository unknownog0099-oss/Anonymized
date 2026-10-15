from __future__ import annotations
import argparse
import json
from .core import ROOT, OUTPUT, DATA, environment, integrity, save_json

def main():
    environment()
    parser = argparse.ArgumentParser(description="ProvGuard reproducible research pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ["verify", "download", "dedup", "curated", "prepare-leakage", "audit-queue", "build", "aggregate", "plot"]:
        sub.add_parser(command)
    training = sub.add_parser("train")
    training.add_argument("--method", default="baseline", choices=["baseline", "curated", "random_real", "provguard_250", "provguard_500", "provguard_1000", "provguard_dolly_250", "provguard_oasst1_250", 'random_real_2024','random_real_7777'])
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--augment-train")
    training.add_argument("--batch", type=int, default=16)
    training.add_argument("--accumulation", type=int, default=1)
    training.add_argument("--resume", action="store_true")
    training.add_argument('--semantic-review')
    scoring = sub.add_parser("score")
    scoring.add_argument("--seeds", type=int, nargs="+", default=[13,42,123])
    matrix = sub.add_parser("matrix")
    matrix.add_argument("--seeds", type=int, nargs="+", default=[13, 42, 123, 2024, 7777])
    matrix.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.command == "verify":
        print(json.dumps(integrity(), indent=2))
    elif args.command=='prepare-leakage':
        from .leakage import prepare
        prepare()
    elif args.command in {"download", "dedup", "curated", "audit-queue", "build"}:
        if args.command in {'dedup','curated'} and (DATA/'leakage_index_v2/metadata.json').exists():
            from .leakage import prepare
            prepare()
            return
        from . import data
        getattr(data, {"audit-queue": "audit_queue", "build": "build_sets"}.get(args.command, args.command))()
    elif args.command == "train":
        from .model import train
        train(args.method, args.seed, args.augment_train, args.batch, args.accumulation, resume=args.resume,semantic_review=args.semantic_review)
    elif args.command == "score":
        from .model import score
        score(args.seeds)
    elif args.command in {"aggregate", "plot"}:
        from .final_analysis import export_results
        export_results(final=args.command=='plot')
    elif args.command == "matrix":
        from .model import train
        plan = []
        paths = {"baseline": None, "curated": DATA / "curated_hn.csv", "random_real": DATA / "random_real_1000.csv", **{f"provguard_{k}": DATA / f"provguard_{k}.csv" for k in [250, 500, 1000]}}
        for method, augmentation in paths.items():
            for seed in args.seeds:
                plan.append({"method": method, "seed": seed, "augmentation": str(augmentation) if augmentation else None})
        save_json(ROOT / "reports/experiment_matrix.json", plan)
        if args.execute:
            from .study import main as finish, SEEDS
            if set(args.seeds)!=set(SEEDS):
                raise ValueError('The corrected full study requires all five predefined seeds; use train for a separately documented individual run.')
            finish(['--resume'])
            return
            # Validate all inputs before spending GPU time on any arm.
            for path in paths.values():
                if path and not path.exists():
                    raise FileNotFoundError(path)
            for run in plan:
                status = OUTPUT / run["method"] / f"seed_{run['seed']}" / "status.json"
                if status.exists() and json.loads(status.read_text())["status"] == "complete":
                    continue
                train(run["method"], run["seed"], run["augmentation"])
        else:
            print(json.dumps(plan, indent=2))

if __name__ == "__main__":
    main()
