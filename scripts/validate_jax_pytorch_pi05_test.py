import numpy as np
import pytest

from scripts import validate_jax_pytorch_pi05 as parity


def test_error_metrics_reports_flattened_cosine_similarity():
    reference = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    candidate = reference * 2.0

    metric = parity.error_metrics(reference, candidate)

    assert metric["cosine_similarity"] == pytest.approx(1.0)


def test_passes_enforces_optional_minimum_cosine():
    metric = {
        "shape_match": True,
        "mean_absolute_error": 1e-5,
        "max_absolute_error": 2e-5,
        "cosine_similarity": 0.9998,
    }

    assert parity._passes(metric, 1e-4, 5e-4)
    assert not parity._passes(metric, 1e-4, 5e-4, minimum_cosine=0.9999)


def test_error_metrics_handles_zero_norm_without_false_cosine_pass():
    metric = parity.error_metrics(np.zeros(4, dtype=np.float32), np.ones(4, dtype=np.float32))

    assert metric["cosine_similarity"] is None
