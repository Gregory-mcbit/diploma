import os
import sys
import json
import subprocess
from typing import List, Dict
from app.observability.logger import get_logger
MODEL_PATH  = "data/models/xgb_alpha.json"
WORKER_PATH = "app/ml/xgb_worker.py"
logger = get_logger(__name__)


def run_ml_scoring_pipeline(
    parquet_path: str,
    universe: List[str],
    macro: Dict[str, float] | None = None,
) -> Dict[str, float]:
    """
    Online inference pipeline.
    Runs XGBoost in an isolated subprocess (avoids Mac ARM OpenMP segfault).

    1. Requires an explicit macro snapshot from the data layer.
    2. Spawns app/ml/xgb_worker.py as a child process.
    3. Passes parquet_path, universe, and macro dict via stdin JSON.
    4. Returns Dict[ticker -> predicted_21d_alpha].
    """
    if macro is None:
        raise ValueError("run_ml_scoring_pipeline requires an explicit macro snapshot.")

    payload = json.dumps({
        "parquet_path": parquet_path,
        "universe":     universe,
        "macro":        macro,
    })

    try:
        result = subprocess.run(
            [sys.executable, WORKER_PATH],
            input=payload,
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ,
                 "KMP_DUPLICATE_LIB_OK": "TRUE",
                 "TOKENIZERS_PARALLELISM": "false"},
        )

        if result.returncode != 0:
            raise RuntimeError(f"ML worker failed: {result.stderr[-1000:]}")

        scores: Dict[str, float] = json.loads(result.stdout.strip())
        for ticker, alpha in scores.items():
            logger.info("Scored %s with predicted return %+0.4f.", ticker, alpha)
        return scores

    except subprocess.TimeoutExpired:
        raise RuntimeError("ML worker timed out after 120s.")
    except Exception as e:
        raise RuntimeError(f"ML scoring pipeline failed: {e}") from e
