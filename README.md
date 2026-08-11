# Voice AI Evaluation Framework

An end-to-end QA evaluation framework for Voice AI pipelines.
Tests ASR accuracy, latency, LLM response quality, and edge cases
across a Deepgram STT + OpenAI LLM pipeline.

Built to demonstrate practical AI QA and evaluation engineering skills
across real-world Voice AI pipeline architectures.

---

## What This Framework Tests

| Layer | What We Test | Tool |
|---|---|---|
| Response Schema | All fields present, correct types, valid ranges | pytest |
| ASR Accuracy | Word Error Rate (WER) against ground truth | jiwer |
| Latency | P90 latency for STT, LLM, and full pipeline | pytest + time |
| Edge Cases | Short audio, Hindi input, hallucination guard, consistency | pytest |
| LLM Quality | Relevancy and hallucination scoring | deepeval |
| Regression | Output stability across model updates | pytest + JSON |

---

## Architecture

Audio File → Deepgram STT (nova-3) → Transcript → OpenAI LLM (gpt-4o-mini) → Response

The framework sits on top of this pipeline and evaluates every layer independently.

---

## Tech Stack

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.11.9 | Core language |
| Deepgram SDK | 4.0.0 | Speech to Text (STT) — nova-3 model |
| OpenAI SDK | 1.97.0 | LLM responses — gpt-4o-mini |
| pytest | 8.3.5 | Test runner and framework |
| jiwer | 3.1.0 | Word Error Rate (WER) calculation |
| deepeval | 4.1.4 | LLM output quality scoring |
| gTTS | 2.5.3 | Audio fixture generation |
| python-dotenv | 1.1.1 | API key management |
| httpx | 0.28.1 | HTTP client for API calls |

---

## Project Structure

```
voice-ai-eval-framework/
├── agent/
│   ├── voice_agent.py          # Core STT + LLM pipeline
│   └── smoke_test.py           # Manual sanity check
├── fixtures/
│   ├── generate_fixtures.py    # Generates audio test files
│   ├── basic_question.mp3      # "What are your business hours?"
│   ├── account_balance.mp3     # "I want to check my account balance"
│   ├── hindi_query.mp3         # Hindi language input
│   ├── complaint.mp3           # "My order has not arrived"
│   └── single_word.mp3         # "Yes"
├── tests/
│   ├── test_schema.py          # Test 1 — Schema validation
│   ├── test_asr_accuracy.py    # Test 2 — WER accuracy
│   ├── test_latency.py         # Test 3 — P90 latency
│   ├── test_edge_cases.py      # Test 4 — Edge cases
│   ├── test_llm_quality.py     # Test 5 — LLM quality (deepeval)
│   └── test_regression.py      # Test 6 — Regression snapshots
├── results/                    # Saved baseline outputs
├── .env                        # API keys (never committed)
├── .gitignore
└── requirements.txt
```

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/rohitpkumar/voice-ai-eval-framework.git
cd voice-ai-eval-framework
```

**2. Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add API keys**
Create a `.env` file in the project root:

DEEPGRAM_API_KEY=your_deepgram_key_here
OPENAI_API_KEY=your_openai_key_here

**5. Generate audio fixtures**
```bash
python fixtures/generate_fixtures.py
```

**6. Run all tests**
```bash
pytest tests/ -v
```

---

## Test Results

### Test 1 — Schema Validation

8 tests | 8 passed | 0 failed

Validates all 6 response fields — type, range, and content.
Confidence score: 0.993 on clean English audio.

### Test 2 — ASR Accuracy (WER)

5 tests | 5 passed | 0 failed
Average WER: 2.50% — production quality

| Audio File | WER | Status |
|---|---|---|
| basic_question.mp3 | 0.00% | ✅ Pass |
| account_balance.mp3 | 0.00% | ✅ Pass |
| complaint.mp3 | 10.00% | ✅ Pass |
| single_word.mp3 | 0.00% | ✅ Pass |

### Test 3 — Latency (P90)

4 tests | 4 passed | 0 failed

| Metric | P90 | Average | Std Dev |
|---|---|---|---|
| STT Latency | 1385ms | 1318ms | 141ms |
| LLM Latency | 2479ms | 1797ms | — |
| Total Pipeline | 2725ms | 2653ms | — |

Note: These are REST API measurements.
Production WebSocket streaming achieves under 500ms end to end.

### Test 4 — Edge Cases

9 tests | 9 passed | 0 failed

| Edge Case | Finding | Status |
|---|---|---|
| Single word audio | Handled cleanly, confidence 0.926 | ✅ Pass |
| Hindi language input | Partial keywords detected, confidence 0.941 | ✅ Pass |
| Out of scope question | Correctly refused, no hallucination | ✅ Pass |
| Repeated same audio | Identical transcript all 3 runs, std dev 0.00000 | ✅ Pass |
| All 5 fixtures pipeline | Every fixture completed under 30 seconds | ✅ Pass |

---

## Key Findings

**1. Deepgram nova-3 achieves 2.50% average WER on clean English audio**
Production quality threshold is under 10%. We are well within it.

**2. Number formatting is a real failure mode**
"12345" was transcribed as "12300And45" — 10% WER on that clip.
For use cases involving order IDs, phone numbers, or account numbers,
dedicated numeric entity test coverage is essential.

**3. STT latency is highly consistent**
Standard deviation of 141ms across 3 runs — very stable.
LLM latency is more variable — dependent on OpenAI server load.

**4. Hindi audio partially understood by en-IN model**
Deepgram en-IN model returned "Account balance check" for a Hindi sentence
with 0.941 confidence. For production Indian language support,
a dedicated multilingual model (nova-3 multilingual or Sarvam AI) is needed.

**5. Text normalisation is mandatory for WER calculation**
Deepgram smart_format adds punctuation — "yes." vs "yes" caused 100% WER.
Stripping punctuation before WER calculation is standard benchmarking practice.

---

## Why This Framework Is Model-Agnostic

The test suite evaluates pipeline behaviour — not specific model outputs.
Swap the API endpoint and run the same tests to get comparable metrics
across any Voice AI pipeline architecture.

---

## Deepgram Models Reference

| Model | Best For | Key Fact |
|---|---|---|
| nova-2 | Stable testing baseline | Previous generation |
| nova-3 | Production transcription | 54% better WER — what we use |
| Flux | Real-time voice agents | Built-in turn detection, sub-500ms |

---

## Author

Rohit Kumar
QA Lead & AI Evaluation Engineer
12+ years in QA | Voice AI Evaluation | LLM Testing
[LinkedIn](https://linkedin.com/in/rohitpkumar) | [GitHub](https://github.com/rohitpkumar)
