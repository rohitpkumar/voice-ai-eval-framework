# test_schema.py
# TEST 1 Schema & Response Validation
#
# The most basic but most important test.
# Before we check accuracy or latency, we verify:
#   - Does the pipeline return a response at all?
#   - Does it contain all expected fields?
#   - Are the field types correct?
#   - Are values within acceptable ranges?
#
# In production voice AI, a missing field means
# a broken UI, a crashed workflow, or a silent failure.
# This test catches that before it reaches users.

import sys
import os
import pytest

# Add project root to Python path so we can import our agent
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.voice_agent import run_pipeline

# Fixtures
# pytest fixtures are reusable setup helpers.
# This one runs our pipeline once and shares the result across all tests.
# This avoids making multiple API calls for the same audio file.

@pytest.fixture(scope="module")
def pipeline_result():
    """
    Run the pipeline once on a basic audio file.
    Share the result across all tests in this file.
    scope="module" means it runs once per file, not once per test.
    """
    result = run_pipeline("fixtures/basic_question.mp3")
    return result


# Tests

def test_pipeline_returns_a_result(pipeline_result):
    """
    Most basic check — did we get anything back at all?
    If this fails, something is fundamentally broken.
    """
    assert pipeline_result is not None


def test_all_required_fields_present(pipeline_result):
    """
    Check every expected field exists in the response.
    Missing fields = broken downstream systems.
    This is equivalent to JSON schema validation.
    """
    required_fields = [
        "transcript",
        "confidence",
        "stt_latency_ms",
        "response",
        "llm_latency_ms",
        "total_latency_ms"
    ]
    for field in required_fields:
        assert field in pipeline_result, f"Missing field: {field}"


def test_transcript_is_a_non_empty_string(pipeline_result):
    """
    Transcript must be a string and must not be empty.
    An empty transcript means STT completely failed — 
    nothing downstream will work correctly.
    """
    assert isinstance(pipeline_result["transcript"], str)
    assert len(pipeline_result["transcript"]) > 0


def test_confidence_is_a_valid_float(pipeline_result):
    """
    Confidence must be a number between 0 and 1.
    Deepgram always returns this — if it's missing or out of range,
    something changed in the API response format.
    """
    confidence = pipeline_result["confidence"]
    assert isinstance(confidence, float)
    assert 0.0 <= confidence <= 1.0


def test_llm_response_is_a_non_empty_string(pipeline_result):
    """
    LLM response must be a string and must not be empty.
    An empty response means OpenAI returned nothing —
    the agent would be completely silent to the user.
    """
    assert isinstance(pipeline_result["response"], str)
    assert len(pipeline_result["response"]) > 0


def test_latency_values_are_positive_numbers(pipeline_result):
    """
    All latency values must be positive numbers.
    A zero or negative latency means our timing code is broken.
    A latency above 30000ms (30 seconds) means something timed out.
    """
    assert pipeline_result["stt_latency_ms"] > 0
    assert pipeline_result["llm_latency_ms"] > 0
    assert pipeline_result["total_latency_ms"] > 0
    assert pipeline_result["total_latency_ms"] < 30000


def test_total_latency_equals_sum_of_parts(pipeline_result):
    """
    total_latency_ms should equal stt + llm latency.
    This checks our own calculation logic is correct.
    Small floating point differences are acceptable — we allow 1ms tolerance.
    """
    expected_total = (
        pipeline_result["stt_latency_ms"] +
        pipeline_result["llm_latency_ms"]
    )
    actual_total = pipeline_result["total_latency_ms"]
    assert abs(actual_total - expected_total) < 1.0, (
        f"Total latency {actual_total} does not match "
        f"sum of parts {expected_total}"
    )


def test_transcript_matches_expected_content(pipeline_result):
    """
    For a known audio input, we expect specific words in the transcript.
    basic_question.mp3 says: 'What are your business hours?'
    We check key words appear in the transcript.
    This catches cases where STT returns something completely wrong.
    """
    transcript = pipeline_result["transcript"].lower()
    assert "business" in transcript or "hours" in transcript, (
        f"Expected 'business' or 'hours' in transcript, got: {transcript}"
    )