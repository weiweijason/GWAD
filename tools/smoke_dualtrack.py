"""Run a dependency-backed synthetic smoke test for GWAD-DualTrack.

Usage:
    python tools/smoke_dualtrack.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dualtrack import GWADDualTrackPipeline


def main():
    pipeline = GWADDualTrackPipeline(
        threshold=0.2, similarity_threshold=0.8, window_size=2
    )
    assert pipeline.process_update(torch.zeros(1, 2), "session-a")[0] == "PASS_BENIGN"
    assert pipeline.process_update(
        torch.tensor([[0.1, 0.0]]), "session-a", ds=0.25
    )[0] == "QUEUED"
    status, queue_id, window = pipeline.process_update(
        torch.tensor([[0.2, 0.0]]), "session-a", ds=-0.5
    )
    assert (status, queue_id, window) == ("READY", 0, (0.25, -0.5))
    print("GWAD-DualTrack smoke test passed")


if __name__ == "__main__":
    main()
