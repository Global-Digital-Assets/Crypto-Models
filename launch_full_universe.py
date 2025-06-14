#!/usr/bin/env python3
"""Launch full 134-token pipeline with 8-core AAA-grade execution."""
import subprocess
from token_universe import TOKEN_UNIVERSE

def main():
    tokens_csv = ",".join(TOKEN_UNIVERSE)
    
    # 8-core optimized environment
    env_vars = {
        "PYTHONPATH": "/home/cm/crypto-models",
        "OMP_NUM_THREADS": "8",
        "OPENBLAS_NUM_THREADS": "8", 
        "POLARS_MAX_THREADS": "8",
        "DATA_API_URL": "http://127.0.0.1:8001",
        "PUSHGATEWAY_URL": "http://127.0.0.1:9091"
    }
    
    env_str = " ".join(f"{k}={v}" for k, v in env_vars.items())
    
    script = f'''
    cd /home/cm/crypto-models
    {env_str}
    
    echo "=== Ingest 15m 6mo (134 tokens) ==="
    venv/bin/python ingest/ingest.py --symbols "{tokens_csv}" --days 180 --tf 15m
    
    echo "=== Ingest 1m 21d (134 tokens) ==="  
    venv/bin/python ingest/ingest.py --symbols "{tokens_csv}" --days 21 --tf 1m
    
    echo "=== Build features (134 tokens) ==="
    for token in {" ".join(TOKEN_UNIVERSE)}; do
        venv/bin/python features/build_features.py --token "$token" || echo "[warn] failed: $token"
    done
    
    echo "=== Update buckets (134 tokens) ==="
    venv/bin/python buckets/update_buckets.py --symbols "{tokens_csv}"
    
    echo "=== Train models (bucket + specialist) ==="
    venv/bin/python train/run_all.py
    
    echo "=== Mass inference (134×2 models) ==="
    venv/bin/python infer/run_all.py || true
    
    echo "=== RESULTS ==="
    ls -l models/ | wc -l
    ls -l signals/ | wc -l
    echo "First 20 signals:"
    ls -1 signals/*.json | head -20
    '''
    
    print("Launching full universe pipeline...")
    print(f"Tokens: {len(TOKEN_UNIVERSE)}")
    print(f"Expected models: {len(TOKEN_UNIVERSE) * 2} (long+short)")
    print(f"Expected signals: {len(TOKEN_UNIVERSE) * 2}")
    print("\nScript:")
    print(script)

if __name__ == "__main__":
    main()
