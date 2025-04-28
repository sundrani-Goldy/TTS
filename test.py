import torch
import os
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import soundfile as sf
import time
import traceback
# Path to your trained model

MODEL_PATH = "/home/ubuntu/Testing/TTS/recipes/ljspeech/xtts_v2/run/training/GPT_XTTS-Marathi-Optimized-Continued-April-24-2025_01+21PM-8fbe98ee"
CONFIG_PATH = f"{MODEL_PATH}/config.json"

print(f"Loading config from {CONFIG_PATH}")
# Check if config file exists
if not os.path.exists(CONFIG_PATH):
    print(f"Config file not found at {CONFIG_PATH}")
    print(f"Files in {MODEL_PATH}:")
    print(os.listdir(MODEL_PATH))
    exit(1)

# Load the config
config = XttsConfig()
config.load_json(CONFIG_PATH)
print("Config loaded successfully")

# Debug: Print tokenizer path from config
print(f"Tokenizer path in config: {config.tokenizer_path if hasattr(config, 'tokenizer_path') else 'Not specified'}")

# Find vocab.json in the directory tree
vocab_files = []
for root, dirs, files in os.walk("/home/ubuntu/Testing/TTS/recipes/ljspeech/xtts_v2"):
    if "vocab.json" in files:
        vocab_files.append(os.path.join(root, "vocab.json"))

print(f"Found vocab.json files: {vocab_files}")

# Specify the parent model directory where the base model's tokenizer might be
PARENT_MODEL_DIR = "/home/ubuntu/Testing/TTS/recipes/ljspeech/xtts_v2/XTTS_v2.0_original_model_files"
TOKENIZER_DIR = "/home/ubuntu/Testing/TTS/recipes/ljspeech/xtts_v2"

# Update config with tokenizer path if not already set
# In your model loading code, ensure proper tokenizer initialization
if not hasattr(config, 'tokenizer_path'):
    config.tokenizer_path = "/home/ubuntu/Testing/TTS/recipes/ljspeech/xtts_v2/run/training/XTTS_v2.0_original_model_files/vocab.json"

# Initialize and load the model
print("Initializing model from config...")
model = Xtts.init_from_config(config)  # Add this line
print("Model initialized")

# Now load the checkpoint
print(f"Loading model from: {MODEL_PATH}")
try:
    # Try with explicit vocab_path
    if hasattr(config, 'tokenizer_path') and os.path.exists(config.tokenizer_path):
        print(f"Loading checkpoint with vocab_path: {config.tokenizer_path}")
        model.load_checkpoint(config, checkpoint_dir=MODEL_PATH, eval=True, vocab_path=config.tokenizer_path)
        model = torch.compile(model, mode="max-autotune")
    else:
        print("Loading checkpoint without specifying vocab_path")
        model.load_checkpoint(config, checkpoint_dir=MODEL_PATH, eval=True)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    traceback.print_exc()
    exit(1)

# Debug: Check if tokenizer is loaded
print(f"Tokenizer initialized: {model.tokenizer is not None}")
if model.tokenizer is None:
    print("ERROR: Tokenizer is None! Trying to initialize tokenizer manually...")
    try:
        from TTS.tts.layers.xtts.tokenizer import XTTSTokenizer
        model.tokenizer = XTTSTokenizer(config.tokenizer_path)
        print("Tokenizer manually initialized")
    except Exception as e:
        print(f"Failed to initialize tokenizer: {e}")
        traceback.print_exc()
        exit(1)

# Move to appropriate device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Moving model to {device}")
model.to(device)

# Reference audio for voice characteristics
speaker_reference = "/home/ubuntu/Testing/TTS/hindi3.wav"
print(f"Using speaker reference: {speaker_reference}")
if not os.path.exists(speaker_reference):
    print(f"Speaker reference file not found at {speaker_reference}")
    exit(1)

# Function to synthesize speech with the fine-tuned voice
def synthesize_speech(text, language="hi"):
    print(f"Starting synthesis for text: '{text}'")
    start_time = time.time()
    try:
        # Check tokenizer before synthesis
        if model.tokenizer is None:
            print("ERROR: Tokenizer is still None before synthesis!")
            return None
        outputs = model.synthesize(
            text=text,
            speaker_wav=speaker_reference,
            language=language,
            temperature=0.5,        # Match Hindi setting
            length_penalty=0.95,    # Match Hindi setting
            repetition_penalty=2.6, # Match Hindi setting
            top_k=10,               # Match Hindi setting
            top_p=0.8,              # Same in both
            config=config,
            speed=0.9,              # Add speed parameter
            enable_text_splitting=True,
            gpt_cond_len=1,         # Reduce from 3 to 1
            gpt_cond_chunk_len=1,   # Reduce from 3 to 1
        )
        print(f"Synthesis completed in {time.time() - start_time:.2f} seconds")
        if "wav" in outputs:
            audio = outputs["wav"]
            print(f"Audio generated with shape: {audio.shape}, min: {audio.min()}, max: {audio.max()}")
            return outputs
        else:
            print(f"No 'wav' key in outputs. Keys: {outputs.keys()}")
            return None
    except Exception as e:
        print(f"Error during synthesis: {e}")
        traceback.print_exc()
        return None

# Example usage
text = """
सोमवारी म्हणजे २१ जानेवारीला? ठीक आहे.  २१ जानेवारीला ६१८९ रुपये जमा करण्याचे तुमचे वचन आहे.  तुम्ही जवळच्या शाखेत जाऊन ही रक्कम जमा करू शकता. धन्यवाद आपला दिवस शुभ जाओ
"""
print("Calling synthesize_speech function...")
outputs = synthesize_speech(text)

if outputs is not None and "wav" in outputs:
    output_path = "output2.wav"
    print(f"Saving audio to {output_path}")
    sf.write(output_path, outputs["wav"], 24000)
    print(f"Audio saved to {output_path}")
    # Print file size
    file_size = os.path.getsize(output_path)
    print(f"File size: {file_size} bytes")
else:
    print("No output to save")