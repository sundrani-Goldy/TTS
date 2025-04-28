import torch
import os
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import soundfile as sf
import time
import traceback

# Path to your trained model
MODEL_PATH = "/home/ubuntu/TTS/recipes/ljspeech/xtts_v2/run/training/GPT_XTTS-Marathi-Optimized-Continued-April-24-2025_01+21PM-8fbe98ee"
CONFIG_PATH = f"{MODEL_PATH}/config.json"

# Explicitly set tokenizer path - use one of the found paths
VOCAB_PATH = "/home/ubuntu/TTS/recipes/ljspeech/xtts_v2/run/training/XTTS_v2.0_original_model_files/vocab.json"

# Ensure tokenizer file exists
if not os.path.exists(VOCAB_PATH):
    print(f"Tokenizer file not found at {VOCAB_PATH}")
    # Try alternative path
    VOCAB_PATH = "/home/ubuntu/Testing/TTS/recipes/ljspeech/xtts_v2/run/training/XTTS_v2.0_original_model_files/vocab.json"
    if not os.path.exists(VOCAB_PATH):
        print(f"Alternative tokenizer file not found at {VOCAB_PATH}")
        exit(1)

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

# Set tokenizer path in config
config.tokenizer_path = VOCAB_PATH
print(f"Tokenizer path set to: {config.tokenizer_path}")

# Initialize and load the model
print("Initializing model from config...")
model = Xtts.init_from_config(config)
print("Model initialized")

# Now load the checkpoint with explicit vocab path
print(f"Loading model from: {MODEL_PATH}")
try:
    print(f"Loading checkpoint with vocab_path: {VOCAB_PATH}")
    model.load_checkpoint(config, checkpoint_dir=MODEL_PATH, eval=True, vocab_path=VOCAB_PATH)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    traceback.print_exc()
    exit(1)

# Verify tokenizer is initialized correctly
if model.tokenizer is None:
    print("ERROR: Tokenizer is None! Initializing tokenizer manually...")
    try:
        from TTS.tts.layers.xtts.tokenizer import XTTSTokenizer
        model.tokenizer = XTTSTokenizer(VOCAB_PATH)
        print("Tokenizer manually initialized")
    except Exception as e:
        print(f"Failed to initialize tokenizer: {e}")
        traceback.print_exc()
        exit(1)

# Check tokenizer functionality
try:
    test_encoding = model.tokenizer.encode("Test text", "en")
    print(f"Tokenizer test successful, encoded: {test_encoding[:5]}...")
except Exception as e:
    print(f"Tokenizer test failed: {e}")
    traceback.print_exc()
    exit(1)

# Move to appropriate device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Moving model to {device}")
model.to(device)

# Reference audio for voice characteristics
speaker_reference = "/home/ubuntu/TTS/hindi3.wav"
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
        if model.tokenizer is None or not hasattr(model.tokenizer, 'encode'):
            print("ERROR: Invalid tokenizer before synthesis! Reinitializing...")
            from TTS.tts.layers.xtts.tokenizer import XTTSTokenizer
            model.tokenizer = XTTSTokenizer(VOCAB_PATH)
        
        outputs = model.synthesize(
            text=text,
            speaker_wav=speaker_reference,
            language=language,
            temperature=0.5,
            length_penalty=0.95,
            repetition_penalty=2.6,
            top_k=10,
            top_p=0.8,
            config=config,
            speed=0.9,
            enable_text_splitting=True,
            gpt_cond_len=1,
            gpt_cond_chunk_len=1,
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