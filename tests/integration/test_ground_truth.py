"""Runs the 5 independent hand-verification metrics and asserts they match the
pipeline's own cleaned-artifact output within tolerance (plan section 28.4).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from verify_ground_truth import TOLERANCE_ABS, WEEK_END, WEEK_START, SKU, pipeline_values
from src.verification.ground_truth import run_all_verifications

ROOT = Path(__file__).resolve().parents[2]


def test_five_hand_verified_metrics_match_pipeline():
    data_dir = ROOT / "data" / "qahwa_saihat"
    manual = {r.metric_name: r.manual_value for r in run_all_verifications(data_dir, WEEK_START, WEEK_END, SKU)}
    pipeline = pipeline_values(data_dir)

    assert set(manual.keys()) == set(pipeline.keys())
    for name, manual_value in manual.items():
        agent_value = pipeline[name]
        assert abs(agent_value - manual_value) <= TOLERANCE_ABS, (
            f"{name}: manual={manual_value} agent={agent_value} exceeds tolerance {TOLERANCE_ABS}"
        )
