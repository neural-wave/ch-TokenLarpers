from __future__ import annotations

import pytest

from pvdetect import benchmark
from pvdetect.benchmark import Timing, project


def test_projection_scales_fit_quadratically_and_prep_linearly():
    pilot = Timing(prep_s=6.0, fit_s=60.0, peak_rss_mb=100)
    # n=24 -> factor 2: (60*4 + 6*2) * repeats 2 * scenarios 3 / 60
    assert project(pilot, 24, 2, 3) == pytest.approx((240 + 12) * 2 * 3 / 60)


def test_full_run_refused_over_budget(monkeypatch, capsys):
    monkeypatch.setattr(benchmark, "run_scenario", lambda *a, **k: ({}, Timing(1.0, 600.0, 100)))
    assert benchmark.main(["--suite", "dev", "--scenarios", "base"]) == 3
    assert "refused" in capsys.readouterr().out


def test_holdout_requires_log():
    with pytest.raises(SystemExit):
        benchmark.main(["--suite", "holdout"])
