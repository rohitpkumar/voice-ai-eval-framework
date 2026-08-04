# test_latency.py
# TEST 3 — Latency Testing
#
# Latency is how long the user waits for a response.
# It is one of the most important metrics in voice AI.
# Nobody wants to speak to an agent that takes 10 seconds to reply.
#
# We measure three things:
#   1. STT Latency  — how long Deepgram takes to transcribe audio
#   2. LLM Latency  — how long OpenAI takes to generate a response
#   3. Total Latency — full end to end time the user actually feels
#
# Key concept — P90 (90th Percentile):
#   We do not just measure one call and call it done.
#   We run multiple calls and look at P90.
#
#   P90 means: 90% of calls completed within this time.
#   Example: P90 = 5000ms means 9 out of 10 calls finished under 5 seconds.
#   The remaining 1 out of 10 might be slower — those are outliers.
#
#   Why P90 and not average?
#   Average hides outliers. If 9 calls take 1 second and 1 call takes 20 seconds,
#   average = 2.9 seconds — looks fine. But 1 in 10 users waited 20 seconds.
#   P90 catches this. P90 = 20 seconds — immediately shows the problem.
#
# Industry targets for voice AI:
#   STT latency    : under 500ms (real-time streaming)
#   LLM latency    : under 1000ms
#   Total latency  : under 1500ms for real-time feel
#
# Our setup uses REST API not WebSocket streaming.
# REST adds significant latency vs production streaming.
# Our numbers will be higher — this is expected and documented.
# In interviews: explain REST vs streaming difference clearly.

import sys
import os
import time
import statistics
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.voice_agent import transcribe_audio, get_llm_response, run_pipeline

# ── Configuration ─────────────────────────────────────────────────────────────
# Number of calls to make for latency measurement
# We use 3 to save API credits while still getting meaningful P90 data
# In production benchmarks you would use 50-100 calls
NUM_CALLS = 3

# Audio file for latency testing — use shortest clean file to isolate latency
# We do not want audio processing time to affect our measurement
TEST_AUDIO = "fixtures/basic_question.mp3"

# Latency thresholds for REST API calls (not streaming)
# These are deliberately generous because we are using REST not WebSocket
# In a real production streaming setup targets would be 3-5x lower
STT_LATENCY_THRESHOLD_MS = 8000    # 8 seconds max for REST STT
LLM_LATENCY_THRESHOLD_MS = 10000   # 10 seconds max for REST LLM
TOTAL_LATENCY_THRESHOLD_MS = 20000 # 20 seconds max for full REST pipeline


# ── Helper: calculate percentile ─────────────────────────────────────────────

def percentile(data: list, p: int) -> float:
    """
    Calculate the Pth percentile of a list of numbers.

    Args:
        data: list of numbers (our latency measurements)
        p: percentile to calculate (90 for P90, 50 for median)

    Returns:
        The value at the Pth percentile

    Example:
        data = [1000, 1200, 5000]
        percentile(data, 90) = 5000
        — 90% of calls completed within 5000ms
    """
    # Sort the data first — percentile requires sorted data
    sorted_data = sorted(data)

    # Calculate index position for this percentile
    # Example: 3 items, P90 → index = 0.90 * (3-1) = 1.8 → rounds to index 2
    index = (p / 100) * (len(sorted_data) - 1)

    # If index is a whole number, return that value directly
    if index.is_integer():
        return sorted_data[int(index)]

    # Otherwise interpolate between the two nearest values
    lower = sorted_data[int(index)]
    upper = sorted_data[int(index) + 1]
    return lower + (upper - lower) * (index - int(index))


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_stt_latency_p90():
    """
    Test: STT (Deepgram) latency P90 is within acceptable range.

    We call Deepgram NUM_CALLS times on the same audio file.
    We collect all latency measurements.
    We calculate P90 — 90% of calls must complete within our threshold.

    Why same audio file each time?
    We want to isolate STT latency — not audio length variation.
    Same file = same processing complexity = fair comparison.
    """
    latencies = []

    print(f"\n  Running {NUM_CALLS} STT calls...")
    print(f"  {'Call':<6} {'Latency':>10}")
    print(f"  {'-'*20}")

    for i in range(NUM_CALLS):
        result = transcribe_audio(TEST_AUDIO)
        latency = result["latency_ms"]
        latencies.append(latency)
        print(f"  {i+1:<6} {latency:>8.0f}ms")

    # Calculate statistics
    p50 = percentile(latencies, 50)  # median — typical call
    p90 = percentile(latencies, 90)  # 90th percentile — our quality gate
    avg = statistics.mean(latencies)  # average — for reference only

    print(f"\n  STT Latency Results:")
    print(f"  Average : {avg:.0f}ms")
    print(f"  P50     : {p50:.0f}ms  (50% of calls faster than this)")
    print(f"  P90     : {p90:.0f}ms  (90% of calls faster than this)")
    print(f"  Threshold: {STT_LATENCY_THRESHOLD_MS}ms")

    # P90 is our quality gate — not average
    assert p90 <= STT_LATENCY_THRESHOLD_MS, (
        f"STT P90 latency {p90:.0f}ms exceeds threshold {STT_LATENCY_THRESHOLD_MS}ms"
    )


def test_llm_latency_p90():
    """
    Test: LLM (OpenAI) latency P90 is within acceptable range.

    We call OpenAI NUM_CALLS times with the same transcript.
    This isolates LLM latency from STT latency.

    We use a fixed transcript string — not a live STT call.
    This means we are measuring pure LLM response time only.
    """
    latencies = []

    # Fixed transcript — same input every call for fair comparison
    test_transcript = "What are your business hours?"

    print(f"\n  Running {NUM_CALLS} LLM calls...")
    print(f"  {'Call':<6} {'Latency':>10}")
    print(f"  {'-'*20}")

    for i in range(NUM_CALLS):
        result = get_llm_response(test_transcript)
        latency = result["latency_ms"]
        latencies.append(latency)
        print(f"  {i+1:<6} {latency:>8.0f}ms")

    # Calculate statistics
    p50 = percentile(latencies, 50)
    p90 = percentile(latencies, 90)
    avg = statistics.mean(latencies)

    print(f"\n  LLM Latency Results:")
    print(f"  Average : {avg:.0f}ms")
    print(f"  P50     : {p50:.0f}ms")
    print(f"  P90     : {p90:.0f}ms")
    print(f"  Threshold: {LLM_LATENCY_THRESHOLD_MS}ms")

    assert p90 <= LLM_LATENCY_THRESHOLD_MS, (
        f"LLM P90 latency {p90:.0f}ms exceeds threshold {LLM_LATENCY_THRESHOLD_MS}ms"
    )


def test_total_pipeline_latency_p90():
    """
    Test: Full pipeline (STT + LLM) P90 latency is within acceptable range.

    This is the number that matters most to the user.
    From audio in → response out — how long does the whole thing take?

    This test runs the complete pipeline NUM_CALLS times.
    """
    latencies = []

    print(f"\n  Running {NUM_CALLS} full pipeline calls...")
    print(f"  {'Call':<6} {'STT':>8} {'LLM':>8} {'Total':>10}")
    print(f"  {'-'*35}")

    for i in range(NUM_CALLS):
        result = run_pipeline(TEST_AUDIO)
        stt = result["stt_latency_ms"]
        llm = result["llm_latency_ms"]
        total = result["total_latency_ms"]
        latencies.append(total)
        print(f"  {i+1:<6} {stt:>6.0f}ms {llm:>6.0f}ms {total:>8.0f}ms")

    # Calculate statistics
    p50 = percentile(latencies, 50)
    p90 = percentile(latencies, 90)
    avg = statistics.mean(latencies)

    print(f"\n  Total Pipeline Latency Results:")
    print(f"  Average : {avg:.0f}ms")
    print(f"  P50     : {p50:.0f}ms")
    print(f"  P90     : {p90:.0f}ms")
    print(f"  Threshold: {TOTAL_LATENCY_THRESHOLD_MS}ms")

    assert p90 <= TOTAL_LATENCY_THRESHOLD_MS, (
        f"Total pipeline P90 latency {p90:.0f}ms "
        f"exceeds threshold {TOTAL_LATENCY_THRESHOLD_MS}ms"
    )


def test_stt_latency_consistency():
    """
    Test: STT latency does not vary wildly between calls.

    A consistent system is a reliable system.
    If one call takes 500ms and the next takes 8000ms — that is unstable.
    Users experience this as random slowness — very frustrating.

    We measure this using standard deviation.
    Standard deviation tells us how spread out the latency values are.
    Low standard deviation = consistent = good.
    High standard deviation = unpredictable = bad.

    Threshold: standard deviation must be under 3000ms
    If it is higher, our latency is too unpredictable.
    """
    latencies = []

    for i in range(NUM_CALLS):
        result = transcribe_audio(TEST_AUDIO)
        latencies.append(result["latency_ms"])

    if len(latencies) < 2:
        pytest.skip("Need at least 2 calls to measure consistency")

    std_dev = statistics.stdev(latencies)
    avg = statistics.mean(latencies)

    print(f"\n  STT Consistency Results:")
    print(f"  Latencies   : {[f'{l:.0f}ms' for l in latencies]}")
    print(f"  Average     : {avg:.0f}ms")
    print(f"  Std Dev     : {std_dev:.0f}ms")
    print(f"  Threshold   : 3000ms std dev max")

    assert std_dev <= 3000, (
        f"STT latency too inconsistent. "
        f"Std dev {std_dev:.0f}ms exceeds 3000ms threshold. "
        f"Latencies were: {latencies}"
    )