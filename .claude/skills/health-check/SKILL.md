---
name: health-check
description: Pre-flight validation before training or backtesting. Checks model config, feature alignment, calibrator freshness, agent locks, and MLflow connectivity. Run at the start of any session or before a long training run.
allowed-tools: Read, Grep, Glob, Bash
---

# Health Check

Run every check below in order. Print a clear PASS / FAIL / WARN for each item. At the end, print a one-line summary: either "All checks passed — safe to proceed" or a list of blockers.

---

## 1. Model config

Check that `checkpoints/signal_model_config.json` exists and contains all required fields.

```bash
python3 -c "
import json, sys
try:
    c = json.load(open('checkpoints/signal_model_config.json'))
    required = ['holdout_start_date', 'feature_columns', 'feature_mean', 'feature_std', 'sequence_length']
    missing = [k for k in required if k not in c]
    if missing:
        print('FAIL — missing keys:', missing)
        sys.exit(1)
    print('PASS — config present, holdout_start_date:', c['holdout_start_date'])
    print('      ', len(c['feature_columns']), 'feature columns, sequence_length:', c['sequence_length'])
except FileNotFoundError:
    print('FAIL — checkpoints/signal_model_config.json not found. Run: uv run python main.py train')
    sys.exit(1)
"
```

---

## 2. Calibrator freshness

Check that `checkpoints/calibration.json` exists and report when it was last modified vs the model config.

```bash
python3 -c "
import os, json
from datetime import datetime

cal_path = 'checkpoints/calibration.json'
cfg_path = 'checkpoints/signal_model_config.json'

if not os.path.exists(cal_path):
    print('FAIL — calibration.json not found. Run: uv run python main.py calibrate')
else:
    cal_mtime = os.path.getmtime(cal_path)
    cfg_mtime = os.path.getmtime(cfg_path) if os.path.exists(cfg_path) else 0
    cal_age_days = (datetime.now().timestamp() - cal_mtime) / 86400
    if cfg_mtime > cal_mtime:
        print('WARN — calibration.json is older than signal_model_config.json. Recalibrate: uv run python main.py calibrate')
    elif cal_age_days > 30:
        print(f'WARN — calibration.json is {cal_age_days:.0f} days old. Consider recalibrating.')
    else:
        print(f'PASS — calibration.json present, {cal_age_days:.0f} days old')
"
```

---

## 3. Feature column alignment

Check that the feature columns in the saved config match what `data/features.py` would produce. Read `data/features.py` and count the columns returned by `FeatureEngineer`. Compare to `len(feature_columns)` in the config.

```bash
python3 -c "
import json, ast, sys

try:
    cfg = json.load(open('checkpoints/signal_model_config.json'))
    saved_cols = cfg['feature_columns']
    print(f'PASS — config has {len(saved_cols)} feature columns')
    print('      First 5:', saved_cols[:5])
    print('      Last 5:', saved_cols[-5:])
except FileNotFoundError:
    print('SKIP — no config to check against')
"
```

Then read `data/features.py` and verify the column list has not grown or shrunk since the config was saved. If the number of columns in the code differs from the config, print:
`WARN — feature count mismatch: config has N, code produces M. Retrain required if you changed features.`

---

## 4. Model checkpoint files

```bash
ls checkpoints/ 2>/dev/null || echo "FAIL — checkpoints/ directory missing"
```

Check that at minimum these files exist:
- `checkpoints/signal_model_config.json`
- `checkpoints/signal_model.weights.h5` or similar (any `.h5` or `.keras` file)

```bash
python3 -c "
import glob, os
weights = glob.glob('checkpoints/*.h5') + glob.glob('checkpoints/*.keras')
if weights:
    sizes = [(os.path.basename(w), f'{os.path.getsize(w)/1e6:.1f} MB') for w in weights]
    print('PASS — model weights found:', sizes)
else:
    print('FAIL — no model weights in checkpoints/. Run: uv run python main.py train')
"
```

---

## 5. Agent locks

Check for stale locks — tasks claimed but potentially abandoned.

```bash
ls .agent-locks/ 2>/dev/null && echo "WARN — active locks exist (see above). Another agent may be working." || echo "PASS — no active locks"
```

If locks exist, read `TASKS.md` and report which task each lock corresponds to. A lock is likely stale if the task status in TASKS.md is still `in_progress` but no recent git activity touches that task's scope files.

---

## 6. MLflow connectivity

```bash
python3 -c "
import os
uri = os.environ.get('MLFLOW_TRACKING_URI', 'not set')
print(f'MLFLOW_TRACKING_URI: {uri}')

import subprocess
result = subprocess.run(['docker', 'ps', '--filter', 'name=mlflow', '--format', '{{.Names}} {{.Status}}'],
                       capture_output=True, text=True)
if result.returncode == 0 and result.stdout.strip():
    print('PASS — MLflow container running:', result.stdout.strip())
else:
    print('WARN — MLflow container not detected. Results will fall back to ./mlruns/. Start with: docker-compose up -d')
"
```

---

## 7. Python environment

```bash
uv run python --version && echo "PASS — uv environment OK" || echo "FAIL — uv environment broken. Run: uv sync"
```

---

## Output format

After running all checks, print:

```
=== Health Check Summary ===
Config:        PASS/FAIL
Calibrator:    PASS/WARN/FAIL
Features:      PASS/WARN
Checkpoints:   PASS/FAIL
Agent locks:   PASS/WARN
MLflow:        PASS/WARN
Environment:   PASS/FAIL

Status: READY / BLOCKED (list blockers)
```

If status is BLOCKED, list the exact commands needed to fix each blocker before proceeding.
