# generate_fixtures.py
# This script creates audio test files (.mp3) that we use in our tests.
# gTTS (Google Text to Speech) converts text to speech for free.
# These audio files simulate what a real user would say to a voice agent.

from gtts import gTTS
import os

# Make sure the fixtures folder exists
os.makedirs("fixtures", exist_ok=True)

# Each fixture is a dict with:
#   - text: what will be spoken in the audio
#   - filename: where to save it
#   - description: why this test case exists

fixtures = [
    {
        "text": "What are your business hours?",
        "filename": "fixtures/basic_question.mp3",
        "description": "Basic customer support question — clean English"
    },
    {
        "text": "I want to check my account balance please.",
        "filename": "fixtures/account_balance.mp3",
        "description": "Common banking query — tests intent understanding"
    },
    {
        "text": "Mujhe apna account balance check karna hai.",
        "filename": "fixtures/hindi_query.mp3",
        "description": "Hindi query — tests non-English audio handling"
        # Expected: Deepgram may struggle here since we set language to en-IN
        # This is intentional — it reveals a real limitation, which is valuable QA insight
    },
    {
        "text": "My order number is 12345 and it has not arrived yet.",
        "filename": "fixtures/complaint.mp3",
        "description": "Customer complaint — tests entity extraction and tone"
    },
    {
        "text": "Yes.",
        "filename": "fixtures/single_word.mp3",
        "description": "Single word response — tests edge case of very short audio"
    }
]

# Generate each audio file
for fixture in fixtures:
    print(f"Generating: {fixture['filename']}")
    print(f"  Text: {fixture['text']}")
    print(f"  Purpose: {fixture['description']}")

    # Create the gTTS object
    # lang='en' for English, lang='hi' for Hindi
    # We use 'en' for all since gTTS handles the text as-is
    tts = gTTS(text=fixture["text"], lang="en", slow=False)

    # Save to file
    tts.save(fixture["filename"])
    print(f"  Saved ✓\n")

print("All fixtures generated successfully.")
print(f"Files saved in: fixtures/")