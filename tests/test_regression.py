# test_regression.py
# TEST 6 — Regression Snapshot Testing
#
# This is the most important test for long-term production quality.
#
# The problem it solves:
#   Voice AI startups update their models frequently.
#   A model update that fixes one thing often breaks another silently.
#   Nobody notices until a user complains.
#
# What regression testing does:
#   1. First run — save the current outputs as a "baseline snapshot"
#   2. Every future run — compare current outputs against the baseline
#   3. If anything changed — flag it immediately
#
# Real world example:
#   Monday: Deepgram updates nova-3 model
#   Tuesday: Your pipeline transcribes "account balance" as "account ballads"
#   Without regression test: nobody notices for days
#   With regression test: CI pipeline fails immediately on Tuesday morning
#
# This is the QA layer that sits between model updates and production.
# Every serious AI startup needs this. Very few have it built properly.
#
# How our implementation works:
#   - Baseline snapshots saved as JSON files in results/ folder
#   - Each snapshot contains: transcript, confidence, latency
#   - On each test run: compare current output against saved baseline
#   - If transcript changed: flag as regression
#   - If confidence dropped significantly: flag as regression
#
# Two modes:
#   MODE 1 — Baseline does not exist yet: save current output as baseline
#   MODE 2 — Baseline exists: compare current output against it

import sys
import os
import json
import pytest
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.voice_agent import transcribe_audio, run_pipeline

# ── Configuration ─────────────────────────────────────────────────────────────

# Where we save baseline snapshots
RESULTS_DIR = "results"

# How much confidence can drop before we flag it as regression
# 0.05 = 5% drop is acceptable, more than that is a regression
CONFIDENCE_TOLERANCE = 0.05

# Audio files we track for regression
TRACKED_FIXTURES = [
    "fixtures/basic_question.mp3",
    "fixtures/account_balance.mp3",
    "fixtures/single_word.mp3",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_baseline_path(audio_file: str) -> str:
    """
    Get the path where baseline snapshot is saved for a given audio file.
    Example: fixtures/basic_question.mp3 → results/baseline_basic_question.json
    """
    filename = os.path.basename(audio_file)           # basic_question.mp3
    name = os.path.splitext(filename)[0]              # basic_question
    return os.path.join(RESULTS_DIR, f"baseline_{name}.json")


def save_baseline(audio_file: str, result: dict) -> str:
    """
    Save current output as baseline snapshot.
    Called on first run when no baseline exists yet.

    Saves:
        - transcript: what Deepgram heard
        - confidence: how confident Deepgram was
        - timestamp: when this baseline was created
        - audio_file: which file this baseline is for
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    baseline = {
        "audio_file": audio_file,
        "transcript": result["transcript"],
        "confidence": result["confidence"],
        "timestamp": datetime.now().isoformat(),
        "note": "Baseline snapshot — do not edit manually"
    }

    path = get_baseline_path(audio_file)
    with open(path, "w") as f:
        json.dump(baseline, f, indent=2)

    print(f"\n  Baseline saved: {path}")
    print(f"  Transcript : {baseline['transcript']}")
    print(f"  Confidence : {baseline['confidence']}")

    return path


def load_baseline(audio_file: str) -> dict:
    """
    Load saved baseline snapshot for a given audio file.
    Returns None if no baseline exists yet.
    """
    path = get_baseline_path(audio_file)

    if not os.path.exists(path):
        return None

    with open(path, "r") as f:
        return json.load(f)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_create_or_verify_baseline_basic_question():
    """
    Regression test for basic_question.mp3

    First run behaviour:
        No baseline exists → run pipeline → save output as baseline → PASS
        Message: "Baseline created. Run tests again to verify regression."

    Subsequent run behaviour:
        Baseline exists → run pipeline → compare against baseline
        If transcript matches → PASS
        If transcript changed → FAIL (regression detected)
        If confidence dropped more than 5% → FAIL (quality regression)
    """
    audio_file = "fixtures/basic_question.mp3"
    baseline = load_baseline(audio_file)
    current = transcribe_audio(audio_file)

    print(f"\n  Audio file : {audio_file}")
    print(f"  Current transcript  : {current['transcript']}")
    print(f"  Current confidence  : {current['confidence']}")

    if baseline is None:
        # First run — save baseline and pass
        save_baseline(audio_file, current)
        print(f"\n  ✅ Baseline created successfully.")
        print(f"  Run tests again to start regression checking.")
        assert True  # First run always passes
    else:
        # Subsequent runs — compare against baseline
        print(f"\n  Baseline transcript : {baseline['transcript']}")
        print(f"  Baseline confidence : {baseline['confidence']}")
        print(f"  Baseline created at : {baseline['timestamp']}")

        # Check 1 — transcript must match exactly
        assert current["transcript"] == baseline["transcript"], (
            f"\n  ❌ REGRESSION DETECTED — Transcript changed!\n"
            f"  Baseline : {baseline['transcript']}\n"
            f"  Current  : {current['transcript']}\n"
            f"  This may indicate a model update changed transcription behaviour."
        )

        # Check 2 — confidence must not drop significantly
        confidence_drop = baseline["confidence"] - current["confidence"]
        assert confidence_drop <= CONFIDENCE_TOLERANCE, (
            f"\n  ❌ REGRESSION DETECTED — Confidence dropped!\n"
            f"  Baseline : {baseline['confidence']}\n"
            f"  Current  : {current['confidence']}\n"
            f"  Drop     : {confidence_drop:.4f} (tolerance: {CONFIDENCE_TOLERANCE})\n"
            f"  This may indicate audio quality or model degradation."
        )

        print(f"\n  ✅ No regression detected.")
        print(f"  Transcript matches baseline exactly.")
        print(f"  Confidence drop: {confidence_drop:.4f} (within tolerance)")


def test_create_or_verify_baseline_account_balance():
    """
    Regression test for account_balance.mp3
    Same logic as above — create baseline on first run,
    compare on subsequent runs.
    """
    audio_file = "fixtures/account_balance.mp3"
    baseline = load_baseline(audio_file)
    current = transcribe_audio(audio_file)

    print(f"\n  Audio file : {audio_file}")
    print(f"  Current transcript  : {current['transcript']}")
    print(f"  Current confidence  : {current['confidence']}")

    if baseline is None:
        save_baseline(audio_file, current)
        print(f"\n  ✅ Baseline created successfully.")
        assert True
    else:
        print(f"\n  Baseline transcript : {baseline['transcript']}")
        print(f"  Baseline confidence : {baseline['confidence']}")

        assert current["transcript"] == baseline["transcript"], (
            f"\n  ❌ REGRESSION DETECTED — Transcript changed!\n"
            f"  Baseline : {baseline['transcript']}\n"
            f"  Current  : {current['transcript']}"
        )

        confidence_drop = baseline["confidence"] - current["confidence"]
        assert confidence_drop <= CONFIDENCE_TOLERANCE, (
            f"\n  ❌ REGRESSION DETECTED — Confidence dropped!\n"
            f"  Drop: {confidence_drop:.4f} (tolerance: {CONFIDENCE_TOLERANCE})"
        )

        print(f"\n  ✅ No regression detected.")


def test_create_or_verify_baseline_single_word():
    """
    Regression test for single_word.mp3
    Single word clips are most sensitive to model changes.
    If "Yes." becomes "Yeah." after a model update — this catches it.
    """
    audio_file = "fixtures/single_word.mp3"
    baseline = load_baseline(audio_file)
    current = transcribe_audio(audio_file)

    print(f"\n  Audio file : {audio_file}")
    print(f"  Current transcript  : {current['transcript']}")
    print(f"  Current confidence  : {current['confidence']}")

    if baseline is None:
        save_baseline(audio_file, current)
        print(f"\n  ✅ Baseline created successfully.")
        assert True
    else:
        print(f"\n  Baseline transcript : {baseline['transcript']}")
        print(f"  Baseline confidence : {baseline['confidence']}")

        assert current["transcript"] == baseline["transcript"], (
            f"\n  ❌ REGRESSION DETECTED — Transcript changed!\n"
            f"  Baseline : {baseline['transcript']}\n"
            f"  Current  : {current['transcript']}"
        )

        confidence_drop = baseline["confidence"] - current["confidence"]
        assert confidence_drop <= CONFIDENCE_TOLERANCE, (
            f"\n  ❌ REGRESSION DETECTED — Confidence dropped!\n"
            f"  Drop: {confidence_drop:.4f} (tolerance: {CONFIDENCE_TOLERANCE})"
        )

        print(f"\n  ✅ No regression detected.")


def test_baseline_files_exist_after_first_run():
    """
    Meta test — verifies that baseline files were actually saved.
    If this fails, it means save_baseline() is broken.
    This test only runs meaningfully after the first run of the above tests.
    """
    print(f"\n  Checking baseline files exist in {RESULTS_DIR}/")

    for audio_file in TRACKED_FIXTURES:
        path = get_baseline_path(audio_file)
        assert os.path.exists(path), (
            f"Baseline file missing: {path}\n"
            f"Run the regression tests once to create baselines."
        )
        # Load and verify it's valid JSON with required fields
        with open(path, "r") as f:
            data = json.load(f)

        required_keys = ["audio_file", "transcript", "confidence", "timestamp"]
        for key in required_keys:
            assert key in data, (
                f"Baseline file {path} missing key: {key}"
            )

        print(f"  ✅ {os.path.basename(path)} — valid")

    print(f"\n  All {len(TRACKED_FIXTURES)} baseline files present and valid.")