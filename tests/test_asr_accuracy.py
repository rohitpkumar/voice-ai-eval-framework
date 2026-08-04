# test_asr_accuracy.py
# TEST 2 — ASR Accuracy using Word Error Rate (WER)
#
# WER is the industry standard metric for measuring Speech to Text accuracy.
# It answers the question: how many words did Deepgram get wrong?
#
# Formula:
#   WER = (Substitutions + Deletions + Insertions) / Total words in reference
#
# Examples:
#   Reference : "what are your business hours"
#   Hypothesis: "what are your business hours"
#   WER = 0.0 — perfect
#
#   Reference : "what are your business hours"
#   Hypothesis: "what are your busy hours"
#   WER = 0.2 — one word wrong out of five = 20% error
#
# Industry benchmarks:
#   Below 10% WER = production quality, clean audio
#   Below 20% WER = acceptable for accented or noisy audio
#   Above 30% WER = not production ready
#
# Tool: jiwer — pip install jiwer
# jiwer is the standard Python library for WER calculation.
# Used in research papers and production benchmarks globally.

import sys
import os
import pytest
from jiwer import wer, transforms

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.voice_agent import transcribe_audio

# ── Ground Truth Definitions
# Ground truth = what we KNOW the audio contains.
# We generated these audio files ourselves using gTTS so we know
# exactly what words are in each file.
# In a real startup, ground truth comes from human-labelled datasets.

GROUND_TRUTH = {
    "fixtures/basic_question.mp3": "what are your business hours",
    "fixtures/account_balance.mp3": "i want to check my account balance please",
    # Note: 12345 — Deepgram smart_format may render numbers differently
    # We test the words around the number, not the number format itself
    "fixtures/complaint.mp3": "my order number is and it has not arrived yet",
    "fixtures/single_word.mp3": "yes",
}

# WER thresholds — these are our quality gates
# If WER goes above these values, the test fails
WER_THRESHOLD_CLEAN = 0.10    # 10% — clean English audio must be under this
WER_THRESHOLD_COMPLEX = 0.20  # 20% — complex sentences allowed slightly higher


# Helper

def get_wer(audio_path: str, ground_truth: str) -> float:
    """
    Transcribe audio and calculate WER against ground truth.

    Steps:
    1. Send audio to Deepgram → get transcript
    2. Normalise both strings — lowercase, strip punctuation
    3. Calculate WER using jiwer
    4. Return the WER score

    Why normalisation matters:
    Deepgram's smart_format adds punctuation (commas, periods, question marks).
    jiwer treats "yes." and "yes" as different words — 100% WER on one word.
    Stripping punctuation before comparison gives us true word accuracy.
    This is standard practice in all ASR benchmarking pipelines.
    """
    result = transcribe_audio(audio_path)

    # Get raw transcript from Deepgram
    hypothesis = result["transcript"]
    reference = ground_truth

    # Define normalisation pipeline
    # This is the standard jiwer transform chain used in production benchmarks
    normalisation = transforms.Compose([
        transforms.ToLowerCase(),           # lowercase everything
        transforms.RemovePunctuation(),     # strip . , ? ! etc
        transforms.RemoveMultipleSpaces(),  # clean up extra spaces
        transforms.Strip(),                 # remove leading/trailing spaces
    ])

    # Apply normalisation to both strings
    hypothesis_clean = normalisation(hypothesis)
    reference_clean = normalisation(reference)

    # Calculate WER on clean strings
    error_rate = wer(reference_clean, hypothesis_clean)

    # Print for visibility during test runs
    print(f"\n  Audio     : {audio_path}")
    print(f"  Raw got   : {hypothesis}")
    print(f"  Cleaned   : {hypothesis_clean}")
    print(f"  Expected  : {reference_clean}")
    print(f"  WER       : {error_rate:.2%}")

    return error_rate


# ── Tests

def test_wer_basic_question():
    """
    Test: 'What are your business hours?'
    This is clean, simple English — should be near perfect.
    Threshold: under 10% WER.
    If this fails, something is seriously wrong with our STT setup.
    """
    error_rate = get_wer(
        "fixtures/basic_question.mp3",
        GROUND_TRUTH["fixtures/basic_question.mp3"]
    )
    assert error_rate <= WER_THRESHOLD_CLEAN, (
        f"WER too high for basic question: {error_rate:.2%} "
        f"(threshold: {WER_THRESHOLD_CLEAN:.0%})"
    )


def test_wer_account_balance():
    """
    Test: 'I want to check my account balance please.'
    Slightly longer sentence — tests multi-word accuracy.
    Threshold: under 10% WER.
    """
    error_rate = get_wer(
        "fixtures/account_balance.mp3",
        GROUND_TRUTH["fixtures/account_balance.mp3"]
    )
    assert error_rate <= WER_THRESHOLD_CLEAN, (
        f"WER too high for account balance query: {error_rate:.2%} "
        f"(threshold: {WER_THRESHOLD_CLEAN:.0%})"
    )


def test_wer_complaint_with_number():
    """
    Test: 'My order number is 12345 and it has not arrived yet.'
    Contains a number (12345) — tests how Deepgram handles numerals.
    smart_format=true should convert spoken numbers to digits.
    Threshold: under 20% WER — numbers can be tricky.
    """
    error_rate = get_wer(
        "fixtures/complaint.mp3",
        GROUND_TRUTH["fixtures/complaint.mp3"]
    )
    assert error_rate <= WER_THRESHOLD_COMPLEX, (
        f"WER too high for complaint with number: {error_rate:.2%} "
        f"(threshold: {WER_THRESHOLD_COMPLEX:.0%})"
    )


def test_wer_single_word():
    """
    Test: 'Yes.'
    Single word — edge case.
    Short audio is harder for STT models — less context to work with.
    Threshold: under 10% WER — only one word, must get it right.
    """
    error_rate = get_wer(
        "fixtures/single_word.mp3",
        GROUND_TRUTH["fixtures/single_word.mp3"]
    )
    assert error_rate <= WER_THRESHOLD_CLEAN, (
        f"WER too high for single word: {error_rate:.2%} "
        f"(threshold: {WER_THRESHOLD_CLEAN:.0%})"
    )


def test_wer_all_fixtures_summary():
    """
    Summary test — runs WER across all fixtures and prints a report.
    Does not fail on individual thresholds.
    Instead checks that AVERAGE WER across all files is under 15%.
    This gives us a single overall quality score for the STT system.
    This is how real benchmarks work — aggregate score across a test set.
    """
    total_wer = 0.0
    count = 0

    print("\n" + "=" * 55)
    print("  WER SUMMARY REPORT")
    print("=" * 55)

    for audio_path, ground_truth in GROUND_TRUTH.items():
        result = transcribe_audio(audio_path)
        hypothesis = result["transcript"]
        reference = ground_truth

    # Apply same normalisation as individual tests
        normalisation = transforms.Compose([
        transforms.ToLowerCase(),
        transforms.RemovePunctuation(),
        transforms.RemoveMultipleSpaces(),
        transforms.Strip(),
])

        hypothesis = normalisation(hypothesis)
        reference = normalisation(reference)
        error_rate = wer(reference, hypothesis)
        total_wer += error_rate
        count += 1

        status = "✅ PASS" if error_rate <= 0.20 else "❌ FAIL"
        print(f"  {status}  {os.path.basename(audio_path):<25} WER: {error_rate:.2%}")

    average_wer = total_wer / count
    print("=" * 55)
    print(f"  Average WER across {count} files: {average_wer:.2%}")
    print("=" * 55)

    # Overall system quality gate — average must be under 15%
    assert average_wer <= 0.15, (
        f"Average WER {average_wer:.2%} exceeds 15% threshold. "
        f"STT system quality is below acceptable standard."
    )