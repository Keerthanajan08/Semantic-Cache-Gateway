"""
Tests for the evaluation methodology in scripts/evaluate_threshold.py, since
the default threshold and model are chosen by it. Uses the fake embedder
so it's fast and deterministic.
"""

import json
from pathlib import Path

import pytest

from scripts import evaluate_threshold as ev
from tests.conftest import fake_embed

ROOT = Path(__file__).resolve().parent.parent


def row(threshold, wrong, hit=0.5, guards=True):
    return ev.SweepRow(threshold, guards, hit_rate=hit, wrong_rate=wrong, precision=1 - wrong)


class TestChoose:
    def test_picks_lowest_threshold_within_budget(self):
        rows = [row(0.6, 0.05), row(0.7, 0.01), row(0.8, 0.0), row(0.9, 0.0)]
        assert ev.choose(rows, max_wrong=0.01).threshold == 0.7

    def test_skips_a_lucky_dip_before_a_cliff(self):
        """0.6 is within budget on its own, but 0.7 is not: 0.6 must not be chosen."""
        rows = [row(0.5, 0.05), row(0.6, 0.0), row(0.7, 0.03), row(0.8, 0.0)]
        assert ev.choose(rows, max_wrong=0.01).threshold == 0.8

    def test_ignores_unguarded_rows(self):
        rows = [row(0.6, 0.0, guards=False), row(0.6, 0.05), row(0.8, 0.0)]
        assert ev.choose(rows, max_wrong=0.01).threshold == 0.8

    def test_falls_back_to_strictest_threshold(self):
        rows = [row(0.6, 0.5), row(0.9, 0.2)]
        assert ev.choose(rows, max_wrong=0.01).threshold == 0.9


class TestModelRanking:
    def make(self, name, embed_ms, hit_rate):
        r = ev.ModelResult(name, embed_ms, [], row(0.7, 0.0, hit=hit_rate))
        return (r, object())

    def test_expected_latency(self):
        r, _ = self.make("m", embed_ms=50, hit_rate=0.6)
        assert ev.expected_latency_ms(r, llm_ms=800) == pytest.approx(50 + 0.4 * 800)

    def test_slower_model_wins_when_it_avoids_enough_llm_calls(self):
        fast = self.make("fast", embed_ms=15, hit_rate=0.50)  # 15 + 400 = 415
        slow = self.make("slow", embed_ms=50, hit_rate=0.63)  # 50 + 296 = 346
        assert ev.pick_best([fast, slow], llm_ms=800)[0].model == "slow"

    def test_fast_model_wins_when_llm_is_cheap(self):
        fast = self.make("fast", embed_ms=15, hit_rate=0.50)  # 15 + 25 = 40
        slow = self.make("slow", embed_ms=50, hit_rate=0.63)  # 50 + 18.5 = 68.5
        assert ev.pick_best([fast, slow], llm_ms=50)[0].model == "fast"


class TestSweep:
    def test_leave_one_out_counts_hits_and_wrong_answers(self):
        intents = {
            "reset": ["reset my password", "reset my password now"],
            "cancel": ["cancel my plan", "cancel my plan now"],
        }
        vectors = {q: fake_embed(q) for qs in intents.values() for q in qs}
        rows = {(r.threshold, r.guards): r for r in ev.sweep(intents, vectors)}

        # At 0.80 each question's nearest neighbour is its own paraphrase (~0.87).
        assert rows[(0.8, True)].hit_rate == 1.0
        assert rows[(0.8, True)].wrong_rate == 0.0
        # At 0.50 nothing is wrong yet ("my" is the only shared word across intents).
        assert rows[(0.5, True)].precision == 1.0
        # At 0.98 nothing is similar enough.
        assert rows[(0.98, True)].hit_rate == 0.0

    def test_cross_intent_match_counts_as_wrong(self):
        intents = {"on": ["turn on 2fa now"], "off": ["turn off 2fa now"], "other": ["refund"]}
        vectors = {q: fake_embed(q) for qs in intents.values() for q in qs}
        rows = {(r.threshold, r.guards): r for r in ev.sweep(intents, vectors)}

        unguarded, guarded = rows[(0.7, False)], rows[(0.7, True)]
        assert unguarded.wrong_rate == pytest.approx(2 / 3)  # on<->off served each other
        assert guarded.wrong_rate == 0.0  # opposite_action guard blocks both


def test_eval_dataset_is_well_formed():
    data = json.loads((ROOT / "eval/intents.json").read_text(encoding="utf-8"))
    intents = {k: v for k, v in data.items() if not k.startswith("_")}
    questions = [q for qs in intents.values() for q in qs]
    assert len(intents) >= 10
    assert all(len(qs) >= 4 for qs in intents.values())
    assert len(questions) == len(set(questions)), "a question is labelled twice"


def test_guard_probes_are_all_blocked():
    """Every documented trap pair must be caught by some guard."""
    from app.guards import veto_reason

    pairs = json.loads((ROOT / "eval/guard_probes.json").read_text(encoding="utf-8"))["pairs"]
    for cached, incoming in pairs:
        assert veto_reason(incoming, cached), f"not blocked: {incoming!r}"


def test_results_json_has_what_the_dashboard_reads():
    rows = [row(0.6, 0.02, hit=0.8), row(0.7, 0.0, hit=0.6), row(0.7, 0.03, hit=0.7, guards=False)]
    fast = ev.ModelResult("fast", embed_ms=10.0, rows=rows, chosen=rows[1])
    slow = ev.ModelResult("slow", embed_ms=90.0, rows=rows, chosen=rows[1])
    probes = [("refund 1234", "refund 5678", 0.8, True, "numbers")]
    data = ev.results_json([slow, fast], fast, probes, 10, 3, 0.01, 800.0)

    assert data["best_model"] == "fast"
    assert [m["model"] for m in data["models"]] == ["fast", "slow"]  # ranked by expected latency
    model = data["models"][0]
    assert model["chosen"]["threshold"] == 0.7
    assert len(model["sweep"]) == 3
    assert model["expected_latency_ms"] == pytest.approx(10 + 0.4 * 800)
    assert data["guard_probes"][0]["blocked_by"] == "numbers"
    json.dumps(data)  # serialisable


def test_committed_results_json_matches_shipped_thresholds():
    """eval/results.json feeds the dashboard; it must agree with the thresholds the app ships."""
    from app.config import RECOMMENDED_THRESHOLDS

    data = json.loads((ROOT / "eval/results.json").read_text(encoding="utf-8"))
    for model in data["models"]:
        if model["model"] in RECOMMENDED_THRESHOLDS:
            assert model["chosen"]["threshold"] == RECOMMENDED_THRESHOLDS[model["model"]]
    assert data["best_model"] == data["models"][0]["model"]
