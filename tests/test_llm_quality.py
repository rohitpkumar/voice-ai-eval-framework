# test_llm_quality.py
# TEST 5 — LLM Response Quality using Deepeval
#
# So far we have tested:
#   - Schema — does the pipeline return the right fields?
#   - WER — does STT transcribe accurately?
#   - Latency — does the pipeline respond fast enough?
#   - Edge cases — does it handle unusual inputs gracefully?
#
# Now we test something harder:
#   Is the LLM response actually GOOD?
#
# "Good" means:
#   - Relevant — does it answer what was asked?
#   - Faithful — does it stick to what it knows?
#   - Not hallucinating — does it avoid making things up?
#
# This is where Deepeval comes in.
# Deepeval is an open source LLM evaluation library.
# It provides metrics that score LLM outputs automatically.
#
# Metrics we use:
#
# 1. AnswerRelevancyMetric
#    Scores how relevant the response is to the input question.
#    Score 0 to 1. Higher is better.
#    Example: Question = "What are your hours?"
#             Response = "We are open 9am to 5pm" → high relevancy
#             Response = "The weather is nice today" → low relevancy
#
# 2. HallucinationMetric
#    Scores how much the response contains made-up information.
#    Score 0 to 1. LOWER is better (0 = no hallucination).
#    This metric needs a "context" — what the model is allowed to know.
#    If response contains facts not in the context → hallucination.
#
# Why this matters for voice AI startups:
#   A customer support agent that makes up prices, policies, or order
#   details is a legal and business risk.
#   Deepeval catches this automatically in CI/CD pipelines.

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.voice_agent import get_llm_response

# Deepeval imports
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, HallucinationMetric

# ── Configuration ─────────────────────────────────────────────────────────────

# Minimum relevancy score — response must be at least this relevant
# Scale: 0 to 1. We require 0.5 minimum — moderate relevancy at least.
RELEVANCY_THRESHOLD = 0.5

# Maximum hallucination score — lower is better
# Scale: 0 to 1. We allow max 0.5 — some uncertainty is acceptable.
HALLUCINATION_THRESHOLD = 0.5

# The system context — what our agent is allowed to know
# In a real product this would be the knowledge base or RAG context
# We keep it simple here — a basic customer support agent context
AGENT_CONTEXT = [
    "You are a customer support agent.",
    "You help customers with general enquiries.",
    "You do not have access to specific order details.",
    "You do not have access to account information.",
    "If you do not know something, say so honestly.",
    "Business hours are Monday to Friday 9am to 6pm IST."
]


# ── Helper ────────────────────────────────────────────────────────────────────

def create_test_case(input_text: str, context: list = None) -> LLMTestCase:
    """
    Create a Deepeval LLMTestCase from an input and get the LLM response.

    LLMTestCase is Deepeval's core object.
    It holds:
        - input: what the user asked
        - actual_output: what the LLM responded
        - context: what the LLM is allowed to know (for hallucination check)

    Args:
        input_text: the question or statement sent to the LLM
        context: list of facts the LLM is allowed to reference

    Returns:
        LLMTestCase ready for Deepeval metric evaluation
    """
    # Get LLM response
    result = get_llm_response(input_text)
    response = result["response"]

    print(f"\n  Input    : {input_text}")
    print(f"  Response : {response[:150]}...")

    return LLMTestCase(
        input=input_text,
        actual_output=response,
        context=context or AGENT_CONTEXT
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_response_relevancy_basic_question():
    """
    Test: Response to a basic question is relevant.

    We ask "What are your business hours?"
    The response must be relevant to this question.
    An irrelevant response — talking about something else entirely —
    would score low on AnswerRelevancyMetric.

    Deepeval uses an LLM judge internally to score relevancy.
    It asks: "Does this response address the question asked?"
    """
    test_case = create_test_case("What are your business hours?")

    metric = AnswerRelevancyMetric(
        threshold=RELEVANCY_THRESHOLD,
        model="gpt-4o-mini",  # Deepeval uses this model as the judge
        include_reason=True    # Tells us WHY it gave this score
    )

    # Evaluate the test case
    metric.measure(test_case)

    print(f"\n  Relevancy Score  : {metric.score:.3f}")
    print(f"  Threshold        : {RELEVANCY_THRESHOLD}")
    print(f"  Reason           : {metric.reason}")

    assert metric.score >= RELEVANCY_THRESHOLD, (
        f"Response not relevant enough.\n"
        f"Score: {metric.score:.3f} (threshold: {RELEVANCY_THRESHOLD})\n"
        f"Reason: {metric.reason}"
    )


def test_response_relevancy_account_query():
    """
    Test: Response to account balance query is relevant.

    "I want to check my account balance"
    Agent should respond relevantly — either asking for details
    or explaining it cannot access account information.
    Both are relevant responses to this input.
    """
    test_case = create_test_case(
        "I want to check my account balance please."
    )

    metric = AnswerRelevancyMetric(
        threshold=RELEVANCY_THRESHOLD,
        model="gpt-4o-mini",
        include_reason=True
    )

    metric.measure(test_case)

    print(f"\n  Relevancy Score  : {metric.score:.3f}")
    print(f"  Threshold        : {RELEVANCY_THRESHOLD}")
    print(f"  Reason           : {metric.reason}")

    assert metric.score >= RELEVANCY_THRESHOLD, (
        f"Response not relevant enough.\n"
        f"Score: {metric.score:.3f} (threshold: {RELEVANCY_THRESHOLD})\n"
        f"Reason: {metric.reason}"
    )


def test_hallucination_out_of_scope():
    """
    Test: Agent does not hallucinate when asked something out of scope.

    We ask about a specific order — something not in the agent context.
    The agent should NOT invent order details.
    It should admit it does not have access.

    HallucinationMetric compares the response against the context.
    If the response contains facts not grounded in the context → hallucination.

    Score: 0 = no hallucination (perfect)
           1 = complete hallucination (terrible)

    We want score BELOW 0.5.
    """
    out_of_scope_question = (
        "What is the status of my order number 12345? "
        "I placed it last Tuesday."
    )

    test_case = create_test_case(
        out_of_scope_question,
        context=AGENT_CONTEXT
    )

    metric = HallucinationMetric(
        threshold=HALLUCINATION_THRESHOLD,
        model="gpt-4o-mini",
        include_reason=True
    )

    metric.measure(test_case)

    print(f"\n  Hallucination Score : {metric.score:.3f}")
    print(f"  Threshold           : below {HALLUCINATION_THRESHOLD}")
    print(f"  Reason              : {metric.reason}")

    # Lower is better for hallucination
    assert metric.score <= HALLUCINATION_THRESHOLD, (
        f"Hallucination score too high.\n"
        f"Score: {metric.score:.3f} (must be below {HALLUCINATION_THRESHOLD})\n"
        f"Reason: {metric.reason}"
    )


def test_response_relevancy_complaint():
    """
    Test: Response to a customer complaint is relevant.

    "My order has not arrived yet"
    Agent should respond with empathy and next steps.
    An irrelevant response would score poorly.
    """
    test_case = create_test_case(
        "My order number is 12345 and it has not arrived yet."
    )

    metric = AnswerRelevancyMetric(
        threshold=RELEVANCY_THRESHOLD,
        model="gpt-4o-mini",
        include_reason=True
    )

    metric.measure(test_case)

    print(f"\n  Relevancy Score  : {metric.score:.3f}")
    print(f"  Threshold        : {RELEVANCY_THRESHOLD}")
    print(f"  Reason           : {metric.reason}")

    assert metric.score >= RELEVANCY_THRESHOLD, (
        f"Response not relevant enough for complaint.\n"
        f"Score: {metric.score:.3f} (threshold: {RELEVANCY_THRESHOLD})\n"
        f"Reason: {metric.reason}"
    )