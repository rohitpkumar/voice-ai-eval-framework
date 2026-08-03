# smoke_test.py
# This is NOT a formal test.
# It's a quick sanity check to confirm our pipeline works
# before we build the full test suite.
# Run it once, verify it works, then we move to real tests.

import sys
import os

# Add project root to path so we can import voice_agent
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.voice_agent import run_pipeline

print("Running smoke test — basic_question.mp3")
print("-" * 50)

result = run_pipeline("fixtures/basic_question.mp3")

print(f"Transcript    : {result['transcript']}")
print(f"Confidence    : {result['confidence']}")
print(f"STT Latency   : {result['stt_latency_ms']} ms")
print(f"LLM Response  : {result['response']}")
print(f"LLM Latency   : {result['llm_latency_ms']} ms")
print(f"Total Latency : {result['total_latency_ms']} ms")
print("-" * 50)
print("Smoke test complete.")