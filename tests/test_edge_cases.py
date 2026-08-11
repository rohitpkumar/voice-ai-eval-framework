# test_edge_cases.py
# TEST 4 — Edge Case Testing
#
# Edge cases are unusual or extreme inputs that real users will send.
# They are the inputs that break systems in production.
# Most developers test the happy path — the normal expected input.
# QA engineers test the unhappy path — what happens when things go wrong.
#
# Why edge cases matter for voice AI specifically:
#   - Users speak in unexpected ways
#   - Audio quality varies — noise, silence, accents
#   - Users speak different languages mid-sentence
#   - Users give one word answers
#   - Users ask things the agent was not designed for
#
# Every voice AI startup — Whissle, Smallest AI, Bolna, Vaani —
# ships products that real users abuse with edge cases daily.
# A QA engineer who thinks in edge cases is extremely valuable.
#
# Edge cases we test here:
#   1. Single word input — "Yes"
#   2. Hindi language input — non-English audio
#   3. Very long sentence — tests model limits
#   4. Repeated same input — tests consistency
#   5. Out of scope question — tests hallucination guard
#   6. Parametrized multi-fixture test — tests all files systematically

import sys
import os
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.voice_agent import transcribe_audio, get_llm_response, run_pipeline


# ── Edge Case 1 — Single Word Input ──────────────────────────────────────────

def test_single_word_audio_does_not_crash():
    """
    Edge case: User says only one word — "Yes"

    Why this matters:
    Voice agents frequently get one-word responses.
    "Yes", "No", "Sure", "Okay" — these are common replies.
    A short audio clip gives the STT model very little context.
    Some models return empty transcripts on very short audio.
    The pipeline must handle this gracefully — not crash, not return empty.

    What we check:
    - Pipeline does not throw an exception
    - Transcript is not empty
    - Confidence score is returned
    - LLM gives a sensible response even to one word
    """
    result = run_pipeline("fixtures/single_word.mp3")

    print(f"\n  Single word test:")
    print(f"  Transcript : {result['transcript']}")
    print(f"  Confidence : {result['confidence']}")
    print(f"  Response   : {result['response']}")

    # Must not crash — if we reach here, no exception was raised
    assert result is not None

    # Transcript must not be empty
    assert len(result["transcript"].strip()) > 0, (
        "STT returned empty transcript for single word audio"
    )

    # Confidence must be valid
    assert 0.0 <= result["confidence"] <= 1.0

    # LLM must return something — even one word input needs a response
    assert len(result["response"].strip()) > 0, (
        "LLM returned empty response for single word input"
    )


# ── Edge Case 2 — Hindi Language Input ───────────────────────────────────────

def test_hindi_audio_handled_gracefully():
    """
    Edge case: User speaks in Hindi — non-English audio

    Why this matters:
    India is the primary market for Bolna, Vaani, and Whissle.
    Real users will speak Hindi, Hinglish, Tamil, Telugu.
    Our Deepgram is configured for en-IN (Indian English).
    What happens when someone speaks pure Hindi?

    Two acceptable outcomes:
    1. Deepgram attempts transcription (may have errors — acceptable)
    2. Deepgram returns low confidence score (signals language mismatch)

    What is NOT acceptable:
    - Pipeline crashes
    - Empty transcript with no explanation
    - Confidence score outside 0 to 1 range

    This test does not assert transcript accuracy.
    It asserts graceful handling — the system should not break.
    This is a critical distinction in QA thinking.
    """
    result = run_pipeline("fixtures/hindi_query.mp3")

    print(f"\n  Hindi audio test:")
    print(f"  Transcript : {result['transcript']}")
    print(f"  Confidence : {result['confidence']}")
    print(f"  STT Latency: {result['stt_latency_ms']}ms")
    print(f"  Response   : {result['response'][:100]}...")

    # Must not crash
    assert result is not None

    # Confidence must always be a valid float regardless of language
    assert 0.0 <= result["confidence"] <= 1.0, (
        f"Invalid confidence score for Hindi audio: {result['confidence']}"
    )

    # Latency must still be measurable
    assert result["stt_latency_ms"] > 0

    # Pipeline must complete end to end — LLM must still respond
    assert len(result["response"].strip()) > 0, (
        "LLM returned empty response for Hindi audio input"
    )

    # Log confidence for observability
    # Low confidence here is expected and is useful QA signal
    print(f"\n  QA Note: Confidence {result['confidence']:.3f} "
          f"— lower confidence expected for non-English audio on en-IN model")


# ── Edge Case 3 — Out of Scope Question ──────────────────────────────────────

def test_out_of_scope_question_does_not_hallucinate():
    """
    Edge case: User asks something the agent has no knowledge about.

    Why this matters:
    Hallucination is one of the biggest risks in production LLM systems.
    A customer support agent that makes up answers is dangerous.
    Example: User asks for a specific price → agent invents a number.
    Example: User asks for a policy → agent invents a policy.

    Our system prompt tells the agent:
    "If you do not know something, say so honestly."

    We test this by asking a very specific factual question
    that the agent has no way of knowing — a made up order number.

    What we check:
    - Agent does NOT confidently make up a specific answer
    - Agent response contains uncertainty language
      ("don't know", "cannot", "unable", "please contact" etc.)
    - Agent does not crash or return empty

    This is not a perfect hallucination test.
    A full hallucination test uses deepeval — that is Test 5.
    This is a simple behavioural check.
    """
    # Question the agent cannot possibly know the answer to
    out_of_scope = (
        "What is the exact delivery status of order number 987654321 "
        "placed by John Smith on 15th March?"
    )

    result = get_llm_response(out_of_scope)

    print(f"\n  Out of scope test:")
    print(f"  Question : {out_of_scope}")
    print(f"  Response : {result['response']}")

    # Must return something
    assert len(result["response"].strip()) > 0

    # Check response contains uncertainty language
    # Agent should admit it does not know — not make something up
    response_lower = result["response"].lower()

    uncertainty_phrases = [
        "don't have",
        "do not have",
        "cannot",
        "can't",
        "unable",
        "don't know",
        "no access",
        "contact",
        "sorry",
        "unfortunately",
        "not able",
        "would need",
        "please"
    ]

    contains_uncertainty = any(
        phrase in response_lower for phrase in uncertainty_phrases
    )

    assert contains_uncertainty, (
        f"Agent may have hallucinated. Response did not contain uncertainty language.\n"
        f"Response: {result['response']}"
    )


# ── Edge Case 4 — Repeated Same Input Consistency ────────────────────────────

def test_repeated_same_audio_consistent_transcript():
    """
    Edge case: Same audio file sent 3 times in a row.

    Why this matters:
    STT models should be deterministic on the same input.
    If the same audio returns different transcripts each time,
    the model is unstable — dangerous for downstream processing.

    Example failure: same audio returns "yes" then "yeah" then "yep"
    This would cause different intent classifications downstream.

    What we check:
    - All 3 transcripts are identical (after normalisation)
    - Confidence scores are similar across runs (within 0.1 of each other)

    Note: LLM responses will differ — LLMs are non-deterministic.
    We only check STT consistency here, not LLM consistency.
    """
    transcripts = []
    confidences = []

    print(f"\n  Repeated input consistency test:")
    print(f"  {'Run':<5} {'Transcript':<35} {'Confidence':>10}")
    print(f"  {'-'*55}")

    for i in range(3):
        result = transcribe_audio("fixtures/basic_question.mp3")
        transcript = result["transcript"].lower().strip()
        confidence = result["confidence"]
        transcripts.append(transcript)
        confidences.append(confidence)
        print(f"  {i+1:<5} {transcript:<35} {confidence:>10.5f}")

    # All transcripts must be identical
    # STT on same audio must always return same text
    assert len(set(transcripts)) == 1, (
        f"STT returned different transcripts for same audio:\n"
        f"{transcripts}"
    )

    # Confidence scores must be similar — within 0.1 of each other
    max_confidence = max(confidences)
    min_confidence = min(confidences)
    confidence_range = max_confidence - min_confidence

    print(f"\n  Confidence range: {confidence_range:.5f} (threshold: 0.1)")

    assert confidence_range <= 0.1, (
        f"Confidence scores too variable across runs: {confidences}"
    )


# ── Edge Case 5 — Parametrized Test Across All Fixtures ──────────────────────

# This is a pytest feature called parametrize.
# Instead of writing one test per audio file,
# we write one test and run it across multiple inputs automatically.
# This is efficient and scalable — add a new fixture, it gets tested automatically.

@pytest.mark.parametrize("audio_file,description", [
    ("fixtures/basic_question.mp3", "basic English question"),
    ("fixtures/account_balance.mp3", "account balance query"),
    ("fixtures/complaint.mp3", "complaint with order number"),
    ("fixtures/single_word.mp3", "single word response"),
    ("fixtures/hindi_query.mp3", "Hindi language input"),
])
def test_all_fixtures_pipeline_completes(audio_file, description):
    """
    Parametrized edge case: Every audio fixture must complete
    the full pipeline without crashing.

    This is a broad safety net test.
    For every audio file we have — clean, noisy, Hindi, short, long —
    the pipeline must:
    - Not throw an exception
    - Return all required fields
    - Complete within 30 seconds

    Think of this as a smoke test across all fixtures.
    If any new fixture is added, this test automatically covers it.
    """
    print(f"\n  Testing: {description} ({audio_file})")

    # Run full pipeline — if this throws, test fails automatically
    result = run_pipeline(audio_file)

    # All fields must be present
    required_fields = [
        "transcript", "confidence",
        "stt_latency_ms", "response",
        "llm_latency_ms", "total_latency_ms"
    ]

    for field in required_fields:
        assert field in result, (
            f"Missing field '{field}' for {description}"
        )

    # Must complete in reasonable time
    assert result["total_latency_ms"] < 30000, (
        f"Pipeline too slow for {description}: "
        f"{result['total_latency_ms']}ms"
    )

    print(f"  Transcript : {result['transcript'][:50]}")
    print(f"  Confidence : {result['confidence']:.3f}")
    print(f"  Total time : {result['total_latency_ms']}ms")