import torch
from TTS.api import TTS

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# List available 🐸TTS models
print(TTS().list_models())

# Init TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# Run TTS
# ❗ Since this model is multi-lingual voice cloning model, we must set the target speaker_wav and language
# Text to speech list of amplitude values as output
wav = tts.tts(text="प्रिय Manju Devi जी, सोनाटा फाइनेंस की ओर से नमस्कार! हम आपको सूचित करना चाहते हैं कि आपके द्वारा लिए गए ऋण की 13 किश्तें, जिनकी कुल राशि ₹3385.00 बकाया है| कृपया इस मामले को गंभीरता से लें और अपनी शेष राशि का आज ही भुगतान करें| धन्यवाद! आपका दिन शुभ हो!", speaker_wav="/home/ubuntu/Testing/TTS/hindi3.wav", language="hi")
# Text to speech to a file
tts.tts_to_file(text="प्रिय Manju Devi जी, सोनाटा फाइनेंस की ओर से नमस्कार! हम आपको सूचित करना चाहते हैं कि आपके द्वारा लिए गए ऋण की 13 किश्तें, जिनकी कुल राशि ₹3385.00 बकाया है| कृपया इस मामले को गंभीरता से लें और अपनी शेष राशि का आज ही भुगतान करें| धन्यवाद! आपका दिन शुभ हो!", speaker_wav="/home/ubuntu/Testing/TTS/hindi3.wav", language="hi", file_path="output.wav")