import subprocess, pathlib, yaml, time, sys
root=pathlib.Path('/root/crypto-models')
py=str(root/'venv/bin/python')
reg=yaml.safe_load((root/'registry.yaml').read_text())
missing=[e['token'] for e in reg['models'] if not (root/'features'/f"{e['token']}.parquet").exists()]
for t in missing:
    print(f'[catchup] {t}', flush=True)
    subprocess.run([py, str(root/'features'/'build_features.py'), '--token', t])
