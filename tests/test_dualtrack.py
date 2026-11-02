import torch
import pytest

from dualtrack import (
    CrossSessionGraphAggregator,
    GWADDualTrackPipeline,
    UpdateVarianceScreener,
)


def test_update_screener_separates_first_small_and_large_updates():
    screener = UpdateVarianceScreener(threshold=0.1)
    first = screener.process(torch.zeros(1, 2), "a")
    small = screener.process(torch.tensor([[0.05, 0.0]]), "a")
    large = screener.process(torch.tensor([[1.0, 0.0]]), "a")

    assert not first.suspicious
    assert small.suspicious
    assert not large.suspicious
    assert small.delta_norm == pytest.approx(0.05)


def test_aggregator_merges_similar_updates_across_sessions():
    aggregator = CrossSessionGraphAggregator(similarity_threshold=0.9)
    first = aggregator.find_or_create_queue("a", torch.tensor([1.0, 0.0]))
    second = aggregator.find_or_create_queue("b", torch.tensor([2.0, 0.1]))

    assert first == second
    assert aggregator.sessions_for_queue(first) == frozenset({"a", "b"})


def test_pipeline_returns_window_when_full():
    pipeline = GWADDualTrackPipeline(
        threshold=0.2, similarity_threshold=0.8, window_size=2
    )
    assert pipeline.process_update(torch.zeros(1, 2), "a")[0] == "PASS_BENIGN"
    assert pipeline.process_update(torch.tensor([[0.1, 0.0]]), "a", 0.25)[0] == "QUEUED"
    status, queue_id, window = pipeline.process_update(
        torch.tensor([[0.2, 0.0]]), "a", -0.5
    )

    assert status == "READY"
    assert queue_id == 0
    assert window == (0.25, -0.5)
