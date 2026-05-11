"""
gwen_rl/utils/log.py
Minimal logging: prints to console + optional CSV file.
No wandb/tensorboard dependency.
"""

import os
import csv
import time


_log_file = None
_writer = None
_start_time = time.time()


def init_logging(log_dir: str, run_name: str = "run"):
    global _log_file, _writer
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, run_name + ".csv")
    _log_file = open(path, "w", newline="", encoding="utf-8")
    _writer = None   # lazy init on first log call
    print("[log] CSV log: " + path)


def log_metrics(step: int, metrics: dict):
    global _writer
    metrics["step"] = step
    metrics["elapsed_s"] = round(time.time() - _start_time, 1)

    # Console
    parts = ["step=" + str(step)]
    for k, v in metrics.items():
        if k in ("step", "elapsed_s"):
            continue
        if isinstance(v, float):
            parts.append(k + "=" + f"{v:.4f}")
        else:
            parts.append(k + "=" + str(v))
    print("[log] " + "  ".join(parts))

    # CSV
    if _log_file is not None:
        if _writer is None:
            _writer = csv.DictWriter(_log_file, fieldnames=list(metrics.keys()))
            _writer.writeheader()
        _writer.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in metrics.items()})
        _log_file.flush()


def close_logging():
    if _log_file is not None:
        _log_file.close()
