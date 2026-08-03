# voice_agent.py
# This is the core of our voice AI pipeline.
# It does two things:
#   1. Takes an audio file → sends to Deepgram → gets back a transcript (STT)
#   2. Takes that transcript → sends to OpenAI → gets back a response (LLM)
# This is exactly what Whissle, Smallest AI, and Bolna do under the hood.

import os
import httpx
from openai import OpenAI
from dotenv import load_dotenv

# Load API keys from .env file
load_dotenv()

# ── STT: Speech to Text ──────────────────────────────────────────────────────

def transcribe_audio(audio_file_path: str) -> dict:
    """
    Send an audio file to Deepgram and get back a transcript.

    Args:
        audio_file_path: path to the .wav or .mp3 audio file

    Returns:
        dict with keys:
            - transcript (str): the text Deepgram heard
            - confidence (float): how confident Deepgram is (0 to 1)
            - latency_ms (float): how long the API call took in milliseconds
    """

    # Get the Deepgram API key from environment
    api_key = os.getenv("DEEPGRAM_API_KEY")

    # Deepgram's REST endpoint for pre-recorded audio
    url = "https://api.deepgram.com/v1/listen"

    # Parameters we send to Deepgram:
    # - model: nova-2 is Deepgram's most accurate model, included in free tier
    # - smart_format: cleans up punctuation and formatting automatically
    # - language: en-IN targets Indian English accent — relevant for our use case
    params = {
        "model": "nova-2",
        "smart_format": "true",
        "language": "en-IN"
    }

    # Headers tell Deepgram who we are and what we're sending
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "audio/wav"
    }

    # Read the audio file as raw bytes
    with open(audio_file_path, "rb") as audio_file:
        audio_data = audio_file.read()

    # Send the request and measure how long it takes
    # This latency measurement is important — it becomes our test metric
    import time
    start_time = time.time()

    response = httpx.post(
        url,
        headers=headers,
        params=params,
        content=audio_data,
        timeout=30.0  # 30 second timeout — if Deepgram takes longer, something is wrong
    )

    end_time = time.time()
    latency_ms = (end_time - start_time) * 1000  # convert seconds to milliseconds

    # Check if Deepgram returned a successful response
    # If not, raise an error with the full response so we can debug it
    if response.status_code != 200:
        raise Exception(f"Deepgram API error {response.status_code}: {response.text}")

    # Parse the JSON response from Deepgram
    result = response.json()

    # Deepgram's response is nested — we dig into it to get what we need
    # Structure: result → results → channels → [0] → alternatives → [0]
    transcript = (
        result["results"]["channels"][0]["alternatives"][0]["transcript"]
    )
    confidence = (
        result["results"]["channels"][0]["alternatives"][0]["confidence"]
    )

    return {
        "transcript": transcript,
        "confidence": confidence,
        "latency_ms": round(latency_ms, 2)
    }


# ── LLM: Language Model Response ─────────────────────────────────────────────

def get_llm_response(transcript: str, system_prompt: str = None) -> dict:
    """
    Send a transcript to OpenAI and get back an agent response.

    Args:
        transcript: the text we got from Deepgram STT
        system_prompt: optional instruction for how the agent should behave
                       defaults to a simple customer support agent

    Returns:
        dict with keys:
            - response (str): the agent's reply
            - latency_ms (float): how long the API call took in milliseconds
    """

    # Default system prompt — defines the agent's personality and role
    # In a real startup this would be their product's core prompt
    if system_prompt is None:
        system_prompt = (
            "You are a helpful customer support agent. "
            "Answer questions clearly and concisely. "
            "If you don't know something, say so honestly — do not make up answers."
        )

    # Initialize OpenAI client — it automatically picks up OPENAI_API_KEY from environment
    client = OpenAI()

    import time
    start_time = time.time()

    # Call OpenAI's chat completion endpoint
    # - model: gpt-4o-mini is the cheapest model, good enough for our testing
    # - max_tokens: 200 keeps responses short and saves credits
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=200,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": transcript}
        ]
    )

    end_time = time.time()
    latency_ms = (end_time - start_time) * 1000

    # Extract the response text from OpenAI's response object
    response_text = completion.choices[0].message.content

    return {
        "response": response_text,
        "latency_ms": round(latency_ms, 2)
    }


# ── Full Pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(audio_file_path: str, system_prompt: str = None) -> dict:
    """
    Run the full voice AI pipeline end to end.
    Audio file → STT → LLM → Response

    This is the function our tests will call.

    Args:
        audio_file_path: path to the audio file
        system_prompt: optional agent personality/instruction

    Returns:
        dict with all results combined:
            - transcript: what Deepgram heard
            - confidence: Deepgram's confidence score
            - stt_latency_ms: time taken by Deepgram
            - response: OpenAI's reply
            - llm_latency_ms: time taken by OpenAI
            - total_latency_ms: full end-to-end time
    """

    # Step 1 — STT
    stt_result = transcribe_audio(audio_file_path)

    # Step 2 — LLM (using transcript from Step 1)
    llm_result = get_llm_response(stt_result["transcript"], system_prompt)

    # Combine everything into one result dict
    return {
        "transcript": stt_result["transcript"],
        "confidence": stt_result["confidence"],
        "stt_latency_ms": stt_result["latency_ms"],
        "response": llm_result["response"],
        "llm_latency_ms": llm_result["latency_ms"],
        "total_latency_ms": round(
            stt_result["latency_ms"] + llm_result["latency_ms"], 2
        )
    }