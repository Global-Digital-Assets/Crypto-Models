#!/usr/bin/env python
"""Bulk token onboarding utility.

Usage examples:
$ python onboard_token.py --tokens WIF,DOGE,BTC --mode short --parallel 4
$ python onboard_token.py --all --mode long --horizon 60 --target-pct 1.0 --parallel 6

The script automates the repetitive pipeline of
1. Building features
2. Training a model
3. Back-testing and extracting 99th-percentile PnL
4. Deploying the model (archive snapshot, live alias, update registry & bucket map) if the
   PnL passes the profitability hurdle.

The profitability hurdle defaults to 0.35 % for shorts and 0.20 % for longs and can be
adjusted via --pnl-thresh.

Deployment replicates the manual steps we used previously:
 • models/<token>_<mode>/  –> archived to models/<token>_<mode>_vYYYYMMDD-HHMM/
 • Live alias models/<token>USDT_<mode>/ created (+DEPLOY_READY)
 • registry.yaml and bucket_mapping.csv appended if the token is new.

Designed to be run from /root/crypto-models. Uses subprocesses to reuse existing CLI
scripts so the codebase does not need to be imported as a library.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None  # Will fall back to naive append

API_BASE = os.environ.get("DATA_API", "http://127.0.0.1:8001")
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
REGISTRY_YAML = PROJECT_ROOT / "registry.yaml"
BUCKET_MAP = PROJECT_ROOT / "bucket_mapping.csv"
BACKTEST_99_RE = re.compile(r"\s+99\s+\d+\s+\+?(-?\d+\.\d+)")


def fetch_all_symbols() -> List[str]:
    """Attempt to fetch symbol list from Data-API. Falls back to empty list."""
    import requests  # local import to avoid hard dep if not available

    try:
        resp = requests.get(f"{API_BASE}/symbols", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        # Expect list[str]
        symbols: List[str] = data if isinstance(data, list) else []
        return symbols
    except Exception:
        return []


def run_cmd(cmd: List[str]) -> str:
    """Run command from PROJECT_ROOT with PYTHONPATH=.
    Returns captured stdout+stderr."""
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(PROJECT_ROOT))
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command {' '.join(cmd)} failed:\n{proc.stdout}")
    return proc.stdout


def parse_backtest_pnl(out: str) -> float:
    """Return 99th-percentile Avg_PnL_% from backtest output."""
    m = BACKTEST_99_RE.search(out)
    if not m:
        return float("nan")
    return float(m.group(1))


def deploy(token: str, mode: str) -> None:
    """Archive and create live alias; update registry and bucket map."""
    model_dir = MODELS_DIR / f"{token}_{mode}"
    if not model_dir.exists():
        print(f"[warn] {model_dir} missing, cannot deploy", file=sys.stderr)
        return

    ts = datetime.utcnow().strftime("%Y%m%d-%H%M")
    archive_dir = MODELS_DIR / f"{token}_{mode}_v{ts}"
    shutil.copytree(model_dir, archive_dir, dirs_exist_ok=True)

    live_name = f"{token}USDT_{mode}"
    live_dir = MODELS_DIR / live_name
    shutil.copytree(model_dir, live_dir, dirs_exist_ok=True)
    (live_dir / "DEPLOY_READY").touch(exist_ok=True)

    # Update registry.yaml
    if REGISTRY_YAML.exists():
        try:
            updated = False
            if yaml:
                reg = yaml.safe_load(REGISTRY_YAML.read_text()) or []
                if not any(item.get("token") == f"{token}USDT" for item in reg):
                    reg.append({"token": f"{token}USDT", "modes": [mode]})
                    REGISTRY_YAML.write_text(yaml.dump(reg, sort_keys=False))
                    updated = True
            else:  # simple append
                with REGISTRY_YAML.open("a") as fh:
                    fh.write(f"  - token: {token}USDT\n    modes: [{mode}]\n")
                    updated = True
            if updated:
                print(f"[+] registry.yaml updated with {token}USDT")
        except Exception as exc:
            print(f"[warn] failed to update registry.yaml: {exc}")

    # Update bucket mapping CSV
    if not BUCKET_MAP.exists() or f"{token}USDT" not in BUCKET_MAP.read_text():
        with BUCKET_MAP.open("a") as fh:
            fh.write(f"{token}USDT,{token}USDT\n")
        print(f"[+] bucket_mapping.csv appended with {token}USDT")

    print(f"[✓] Deployed {live_name}")


def process_token(token: str, args: argparse.Namespace) -> Tuple[str, float]:
    """Process a single token and return (token, pnl99)."""

    print(f"[ ] Processing {token}")
    prefix = [sys.executable, "-m"]  # use same Python
    try:
        # Step 1 build features
        run_cmd(prefix + ["features.build_features", "--token", token])
        # Step 2 train
        run_cmd(prefix + [
            "train.train",
            token,
            "--mode", args.mode,
            "--horizon", str(args.horizon),
            "--target-pct", str(args.target_pct),
            "--window", str(args.window),
        ])
        # Step 3 backtest
        bt_out = run_cmd(prefix + [
            "backtest",
            "--model", f"{token}_{args.mode}",
            "--percentiles", "90,95,97,99",
            "--tp", "6.0",
            "--sl", "1.5",
            "--window", str(args.window),
        ])
        pnl99 = parse_backtest_pnl(bt_out)
        print(f"[ ] {token} 99th-pct PnL = {pnl99:.4f}%")

        if pnl99 >= args.pnl_thresh:
            deploy(token, args.mode)
        else:
            print(f"[-] {token} below threshold ({pnl99:.2f}% < {args.pnl_thresh}%) – shadow only")
        return token, pnl99
    except Exception as exc:
        print(f"[x] {token} failed: {exc}")
        return token, float("nan")


def main() -> None:
    p = argparse.ArgumentParser(description="Bulk onboard tokens into crypto-models pipeline.")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--tokens", help="Comma-separated list of tokens to process")
    grp.add_argument("--all", action="store_true", help="Process every token returned by Data-API /symbols")
    p.add_argument("--mode", choices=["short", "long"], default="short")
    p.add_argument("--horizon", type=int, default=240, help="Target horizon in minutes")
    p.add_argument("--target-pct", type=float, default=2.0, help="Future return threshold, e.g. 2.0 for ±2 %")
    p.add_argument("--window", type=int, default=30, help="Back-test window in days")
    p.add_argument("--parallel", type=int, default=os.cpu_count() or 4, help="Parallel workers")
    p.add_argument("--pnl-thresh", type=float, default=0.35, help="Deploy if 99-pct PnL ≥ this")

    args = p.parse_args()

    if args.mode == "long" and args.pnl_thresh == 0.35:
        # sensible default lower hurdle for longs
        args.pnl_thresh = 0.20

    if args.all:
        tokens = fetch_all_symbols()
        if not tokens:
            print("[error] Could not retrieve symbols list from Data-API; specify --tokens instead", file=sys.stderr)
            sys.exit(1)
    else:
        tokens = [t.strip() for t in args.tokens.split(",") if t.strip()]
    tokens = sorted(set(tokens))

    print(f"Processing {len(tokens)} tokens, mode={args.mode}, parallel={args.parallel}")
    from functools import partial
    worker = partial(process_token, args=args)
    with cf.ProcessPoolExecutor(max_workers=args.parallel) as ex:
        list(ex.map(worker, tokens))

    # Restart inference timer so new models start generating signals.
    try:
        subprocess.run(["systemctl", "restart", "crypto-models-infer.timer"], check=False)
    except Exception:
        pass


if __name__ == "__main__":
    main()
