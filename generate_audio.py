"""
generate_audio.py — Generate the meme audio file using gTTS

Run once to create audio/tienes_que_tlabajal.mp3
Requires: pip install gTTS
"""

from pathlib import Path

from gtts import gTTS

AUDIO_DIR = Path(__file__).parent / "audio"
AUDIO_DIR.mkdir(exist_ok=True)

OUTPUT = AUDIO_DIR / "tienes_que_tlabajal.mp3"

print("Generating audio...")
tts = gTTS(text="TIENES QUE TRABAJAR", lang="es", slow=False)
tts.save(str(OUTPUT))
print(f"Saved → {OUTPUT}")
