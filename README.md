# Voice AI Evaluation Framework

An end-to-end QA evaluation framework for Voice AI pipelines.
Tests ASR accuracy, latency, LLM response quality, edge cases,
and regression stability across a Deepgram STT + OpenAI LLM pipeline.

Built to demonstrate practical AI QA and evaluation engineering skills
across real-world Voice AI pipeline architectures.

---

## What This Framework Tests

| # | Test Suite | What We Test | Tool |
|---|---|---|---|
| 1 | Schema Validation | All fields present, correct types, valid ranges | pytest |
| 2 | ASR Accuracy | Word Error Rate (WER) against ground truth | jiwer |
| 3 | Latency | P90 latency for STT, LLM, and full pipeline | pytest + time |
| 4 | Edge Cases | Short audio, Hindi input, hallucination guard, consistency | pytest |
| 5 | LLM Quality | Relevancy and hallucination scoring | deepeval |
| 6 | Regression | Output stability across model updates | pytest + JSON |

---

## Architecture

```
Audio File → Deepgram STT (nova-3) → Transcript → OpenAI LLM (gpt-4o-mini) → Response
```

## Tech Stack

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.11.9 | Core language |
| Deepgram SDK | 4.0.0 | Speech to Text — nova-3 model |
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
├── results/
│   ├── baseline_basic_question.json
│   ├── baseline_account_balance.json
│   └── baseline_single_word.json
├── .env                        # API keys (never committed)
├── .gitignore
└── requirements.txt
```

---

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

```
DEEPGRAM_API_KEY=your_deepgram_key_here
OPENAI_API_KEY=your_openai_key_here
```

**5. Generate audio fixtures**

```bash
python fixtures/generate_fixtures.py
```

**6. Run all tests**

```bash
pytest tests/ -v
```

Expected output:

```
34 passed in 95.62s
```

---

## Test 1 — Schema Validation

**What it checks:**
Before testing accuracy or speed, we verify the pipeline returns
what it is supposed to return. Every field. Every time.

**Fields validated:**

```
transcript       → string, non-empty
confidence       → float, between 0.0 and 1.0
stt_latency_ms   → positive number
response         → string, non-empty
llm_latency_ms   → positive number
total_latency_ms → positive number, under 30000
```

**Sample output:**

```
transcript      : What are your business hours?
confidence      : 0.99316
stt_latency_ms  : 4278.0
response        : Our business hours are Monday to Friday...
llm_latency_ms  : 2781.03
total_latency_ms: 7059.03
```

**Result:** 8 tests — 8 passed — 0 failed

---

## Test 2 — ASR Accuracy (WER)

**What it checks:**
Word Error Rate — how many words Deepgram gets wrong as a percentage.

**Formula:**

```
WER = Wrong words / Total words spoken

Example:
You said  : "what are your business hours"  (5 words)
Got back  : "what are your business hours"  (0 wrong)
WER       : 0.00% — perfect
```

**Industry benchmarks:**

```
0–5%    → Excellent, production ready
5–10%   → Good, acceptable
10–20%  → Acceptable for noisy or accented audio
30%+    → Not production ready
```

**Our results:**

| Audio File | WER | Status |
|---|---|---|
| basic_question.mp3 | 0.00% | ✅ Pass |
| account_balance.mp3 | 0.00% | ✅ Pass |
| complaint.mp3 | 10.00% | ✅ Pass |
| single_word.mp3 | 0.00% | ✅ Pass |
| **Average** | **2.50%** | **✅ Production quality** |

**Key finding:**
Number "12345" was transcribed as "12300And45" — 10% WER.
For use cases involving order IDs or phone numbers,
dedicated numeric entity test coverage is essential.

**Result:** 5 tests — 5 passed — 0 failed

---

## Test 3 — Latency (P90)

**What it checks:**
How long users wait for a response. We measure P90 — not average.

**What is P90?**

```
P90 = 90th percentile

Example: P90 = 1385ms means 9 out of 10 calls finished under 1385ms.

Why not average?
9 calls at 500ms + 1 call at 9000ms = average 1350ms (looks fine)
But P90 = 9000ms — shows 1 in 10 users waited 9 seconds.
P90 catches what average hides.
```

**Our results:**

| Metric | P90 | Average | Std Dev |
|---|---|---|---|
| STT Latency (Deepgram) | 1385ms | 1318ms | 141ms |
| LLM Latency (OpenAI) | 2479ms | 1797ms | — |
| Total Pipeline | 2725ms | 2653ms | — |

**Note:**
These are REST API measurements.
Production WebSocket streaming achieves under 500ms end to end.

**Result:** 4 tests — 4 passed — 0 failed

---

## Test 4 — Edge Cases

**What it checks:**
Unusual inputs that real users send — the inputs that break
systems in production.

**1. Single word audio**

```
Input      : Yes.
Confidence : 0.926
Result     : ✅ Handled cleanly — no crash, valid response
```

**2. Hindi language input**

```
Input      : "Mujhe apna account balance check karna hai"
Got back   : "Account balance check"
Confidence : 0.941
Result     : ✅ Partial keywords detected — graceful handling
Finding    : en-IN model partially understands Hindi keywords
```

**3. Out of scope question — hallucination guard**

```
Asked    : "What is delivery status of order 987654321?"
Got back : "I do not have access to specific order details.
            Please check your email confirmation..."
Result   : ✅ Correctly refused — no hallucination
```

**4. Repeated same audio — consistency**

```
Run 1 : "What are your business hours?" — confidence 0.99316
Run 2 : "What are your business hours?" — confidence 0.99316
Run 3 : "What are your business hours?" — confidence 0.99316
Range : 0.00000
Result: ✅ Perfect consistency — STT is deterministic
```

**5. All 5 fixtures parametrized**

```
Every audio file completed full pipeline under 30 seconds.
Fastest : single_word.mp3  — 1843ms
Slowest : complaint.mp3    — 3126ms
```

**Result:** 9 tests — 9 passed — 0 failed

---

## Test 5 — LLM Quality (deepeval)

**What it checks:**
Whether the LLM response is actually good — relevant, faithful,
and free of hallucination.

**Metrics used:**

```
AnswerRelevancyMetric  — does response answer what was asked?
                         Score 0 to 1. Higher is better.
                         Threshold: above 0.5

HallucinationMetric    — does response contain made-up information?
                         Score 0 to 1. Lower is better.
                         Threshold: below 0.5
```

**Our results:**

| Test | Score | Threshold | Status |
|---|---|---|---|
| Business hours relevancy | 0.750 | above 0.5 | ✅ Pass |
| Account query relevancy | 0.750 | above 0.5 | ✅ Pass |
| Out of scope hallucination | 0.167 | below 0.5 | ✅ Pass |
| Complaint relevancy | 1.000 | above 0.5 | ✅ Pass |

**Sample — perfect relevancy score:**

```
Input    : My order number is 12345 and it has not arrived yet.
Response : I am sorry to hear that. Please check the tracking
           information in your confirmation email...
Score    : 1.000
Reason   : Response directly addresses the order inquiry
           without any irrelevant statements.
```

**Sample — hallucination test:**

```
Input    : What is the status of order 12345 placed last Tuesday?
Response : I do not have access to order details or statuses.
           Please check your email confirmation for updates.
Score    : 0.167
Reason   : Response aligns with agent context — correctly
           refused to invent order information.
```

**Result:** 4 tests — 4 passed — 0 failed

---

## Test 6 — Regression Snapshots

**What it checks:**
Whether STT output changes after a model update.
Catches silent regressions before they reach users.

**How it works:**

```
First run  → Save current output as baseline JSON
Future runs → Compare current output against saved baseline
Transcript changed?       → FAIL (regression detected)
Confidence dropped > 5%?  → FAIL (quality regression)
```

**Baseline snapshot example:**

```json
{
  "audio_file": "fixtures/basic_question.mp3",
  "transcript": "What are your business hours?",
  "confidence": 0.99316406,
  "timestamp": "2026-08-14T11:35:50.315019",
  "note": "Baseline snapshot — do not edit manually"
}
```

**Our results:**

| Audio File | Confidence Drop | Transcript Match | Status |
|---|---|---|---|
| basic_question.mp3 | 0.0000 | ✅ Exact match | ✅ No regression |
| account_balance.mp3 | 0.0000 | ✅ Exact match | ✅ No regression |
| single_word.mp3 | 0.0000 | ✅ Exact match | ✅ No regression |

**What a regression failure looks like:**

```
❌ REGRESSION DETECTED — Transcript changed!
Baseline : What are your business hours?
Current  : What are your business hours
Indicates a model update changed transcription behaviour.
```

**Result:** 4 tests — 4 passed — 0 failed

---

> **Note:** All test results are from a local test environment
> using REST API calls. Actual numbers may vary based on network
> conditions, API server load, and model version updates.

## Final Test Scorecard

| Test Suite | Tests | Passed | Failed |
|---|---|---|---|
| Test 1 — Schema Validation | 8 | 8 | 0 |
| Test 2 — ASR Accuracy WER | 5 | 5 | 0 |
| Test 3 — Latency P90 | 4 | 4 | 0 |
| Test 4 — Edge Cases | 9 | 9 | 0 |
| Test 5 — LLM Quality | 4 | 4 | 0 |
| Test 6 — Regression Snapshots | 4 | 4 | 0 |
| **Total** | **34** | **34** | **0** |

```
34 passed in 95.62s — 100% pass rate
```

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
QA Lead & AI Evaluation Engineer | 12+ Years in QA

[LinkedIn](https://linkedin.com/in/rohit-kumar-b7356aa8) | [GitHub](https://github.com/rohitpkumar)
