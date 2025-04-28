import os

from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig
from TTS.utils.manage import ModelManager

import torch
print("CUDA Available:", torch.cuda.is_available())
print("Device Count:", torch.cuda.device_count())
print("Current Device:", torch.cuda.current_device() if torch.cuda.is_available() else "CPU")

# Path to the checkpoint to continue training from
CHECKPOINT_PATH = "/home/ubuntu/Testing/TTS/recipes/ljspeech/xtts_v2/run/training/GPT_XTTS-Marathi-Optimized-April-24-2025_04+59AM-8fbe98ee/checkpoint_35000.pth"

# Logging parameters
RUN_NAME = "GPT_XTTS-Marathi-Optimized-Continued"
PROJECT_NAME = "Marathi-XTTS"
DASHBOARD_LOGGER = "tensorboard"
LOGGER_URI = None

# Set here the path that the checkpoints will be saved. Default: ./run/training/
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run", "training")

# Training Parameters - Optimized for A100 40GB
OPTIMIZER_WD_ONLY_ON_WEIGHTS = True  # for multi-gpu training please make it False
START_WITH_EVAL = False  # if True it will start with evaluation
BATCH_SIZE = 14  # Increased for A100 40GB
GRAD_ACUMM_STEPS = 30  # 20 * 40 = 800 effective batch size
# This effective batch size of 800 provides excellent stability while keeping memory usage manageable

# Define here the dataset that you want to use for the fine-tuning on.
config_dataset = BaseDatasetConfig(
    formatter="ljspeech",
    dataset_name="ljspeech",
    path="/home/ubuntu/Testing/TTS/recipes/ljspeech/xtts_v2/",
    meta_file_train="cleaned_updated_updated.csv",
    language="hi",  # Marathi (ISO 639-1 code)
)

# Add here the configs of the datasets
DATASETS_CONFIG_LIST = [config_dataset]

# Define the path where XTTS v2.0.1 files will be downloaded
CHECKPOINTS_OUT_PATH = os.path.join(OUT_PATH, "XTTS_v2.0_original_model_files/")
os.makedirs(CHECKPOINTS_OUT_PATH, exist_ok=True)


# DVAE files
DVAE_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth"
MEL_NORM_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth"

# Set the path to the downloaded files
DVAE_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(DVAE_CHECKPOINT_LINK))
MEL_NORM_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(MEL_NORM_LINK))

# download DVAE files if needed
if not os.path.isfile(DVAE_CHECKPOINT) or not os.path.isfile(MEL_NORM_FILE):
    print(" > Downloading DVAE files!")
    ModelManager._download_model_files([MEL_NORM_LINK, DVAE_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True)


# Download XTTS v2.0 checkpoint if needed
TOKENIZER_FILE_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/vocab.json"
XTTS_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/model.pth"

# XTTS transfer learning parameters: You we need to provide the paths of XTTS model checkpoint that you want to do the fine tuning.
TOKENIZER_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(TOKENIZER_FILE_LINK))  # vocab.json file
XTTS_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(XTTS_CHECKPOINT_LINK))  # model.pth file

# download XTTS v2.0 files if needed
if not os.path.isfile(TOKENIZER_FILE) or not os.path.isfile(XTTS_CHECKPOINT):
    print(" > Downloading XTTS v2.0 files!")
    ModelManager._download_model_files(
        [TOKENIZER_FILE_LINK, XTTS_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True
    )


# Training sentences generations - using the same reference speakers for better voice cloning
SPEAKER_REFERENCE = [
    "/home/ubuntu/Testing/TTS/recipes/ljspeech/xtts_v2/wavs/marathi_1.wav",
    "/home/ubuntu/Testing/TTS/recipes/ljspeech/xtts_v2/wavs/marathi_2.wav",
    "/home/ubuntu/Testing/TTS/recipes/ljspeech/xtts_v2/wavs/marathi_3.wav",
    "/home/ubuntu/Testing/TTS/recipes/ljspeech/xtts_v2/wavs/marathi_4.wav",
    "/home/ubuntu/Testing/TTS/recipes/ljspeech/xtts_v2/wavs/marathi_5.wav",
    "/home/ubuntu/Testing/TTS/recipes/ljspeech/xtts_v2/wavs/marathi_6.wav",
    "/home/ubuntu/Testing/TTS/recipes/ljspeech/xtts_v2/wavs/marathi_7.wav",
    "/home/ubuntu/Testing/TTS/recipes/ljspeech/xtts_v2/wavs/marathi_8.wav",
    "/home/ubuntu/Testing/TTS/recipes/ljspeech/xtts_v2/wavs/marathi_9.wav",
    "/home/ubuntu/Testing/TTS/recipes/ljspeech/xtts_v2/wavs/marathi_10.wav",
    "/home/ubuntu/Testing/TTS/recipes/ljspeech/xtts_v2/wavs/marathi_11.wav",
    "/home/ubuntu/Testing/TTS/hindi3.wav"
]
LANGUAGE = config_dataset.language

def main():
    # init args and config
    model_args = GPTArgs(
        max_conditioning_length=176400,  # Increased to 8 secs for better voice characteristics capture
        min_conditioning_length=44100,  # 2 secs - still enough for voice identity
        debug_loading_failures=True,  # Set to true to debug any loading failures
        max_wav_length=286000,  # ~13 seconds - increased for better long-form training
        max_text_length=250,    # Increased from 200 to support longer text prompts
        mel_norm_file=MEL_NORM_FILE,
        dvae_checkpoint=DVAE_CHECKPOINT,
        xtts_checkpoint=XTTS_CHECKPOINT,  # checkpoint path of the model that you want to fine-tune
        tokenizer_file=TOKENIZER_FILE,
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        # gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
        # Removed the separate_attention_mask parameter that was causing the error
    )
    
    # define audio config - optimized for quality
    audio_config = XttsAudioConfig(
        sample_rate=22050,  # Original sample rate
        dvae_sample_rate=22050,
        output_sample_rate=24000,  # Higher output sample rate for better quality
    )
    
    # training parameters config
    config = GPTTrainerConfig(
        output_path=OUT_PATH,
        model_args=model_args,
        run_name=RUN_NAME,
        project_name=PROJECT_NAME,
        run_description="""
            GPT XTTS training continued from checkpoint 35000, optimized for voice cloning quality and low latency on A100 40GB
            """,
        dashboard_logger=DASHBOARD_LOGGER,
        logger_uri=LOGGER_URI,
        audio=audio_config,
        batch_size=BATCH_SIZE,
        batch_group_size=48,
        eval_batch_size=BATCH_SIZE,
        num_loader_workers=12,  # Increased for faster data loading on powerful GCP instance
        eval_split_max_size=256,
        print_step=50,
        plot_step=100,
        log_model_step=1000,
        save_step=2500,  # Balanced checkpoint frequency
        save_n_checkpoints=1,  # Keep more checkpoints to select the best one
        save_checkpoints=True,
        mixed_precision=True,  # Enable mixed precision for faster training on A100
        precision="bf16",      # Use bf16 precision to leverage A100 tensor cores
        print_eval=True,  # Enable printing evaluation results
        optimizer="AdamW",
        optimizer_wd_only_on_weights=OPTIMIZER_WD_ONLY_ON_WEIGHTS,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=1e-7,  # Reduced learning rate for continuing from checkpoint
        lr_scheduler="MultiStepLR",
        lr_scheduler_params={"milestones": [40000, 80000, 120000], "gamma": 0.5, "last_epoch": 35000},  # Updated starting epoch
        # Test sentences in Marathi with multiple speakers for comparison
        test_sentences=[
            {
                "text": "छान आहे!",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
            {
                "text": "हा केक फार स्वादिष्ट आहे",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
            {
                "text": "माझ्या आवाजात बोलायला मला खूप आवडतं",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
            {
                "text": "माझा आवाज तयार व्हायला वेळ लागला, पण आता मी गप्प बसणार नाही",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
            {
                "text": "तुमचं ऐकणं खरंच प्रेरणादायक आहे, आणि तुमचा आवाज खूप आत्मविश्वास देतो",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
            {
                "text": "नमस्कार , मी कोटक बँक मधून बोलत आहे , माझे बोलने goldy sundrani यांच्या शी होत आहे का",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
            {
                "text": "नमस्कार , मी HDFC बँक मधून बोलत आहे , माझे बोलने utkarsh mukkawar यांच्या शी होत आहे का",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
            {
                "text": "नमस्कार , मी Axis बँक मधून बोलत आहे , माझे बोलने goldy sundrani यांच्या शी होत आहे का",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
            {
                "text": "नमस्कार , मी SBI बँक मधून बोलत आहे , माझे बोलने goldy sundrani यांच्या शी होत आहे का",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
            {
                "text": "नमस्कार , मी ICICI बँक मधून बोलत आहे , माझे बोलने goldy sundrani यांच्या शी होत आहे का",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
            {
                "text": "नमस्कार , मी रुपामाता मल्टीस्टेट को -ऑप. क्रेडीट सोसायटी लि. मधून बोलत आहे , माझे बोलने mohite shivraj ram यांच्या शी होत आहे का ?",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
        ],
        eval_split_size=0.02,  # Reduced to focus more on training
    )

    # init the model from config
    model = GPTTrainer.init_from_config(config)

    # load training samples
    train_samples, eval_samples = load_tts_samples(
        DATASETS_CONFIG_LIST,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    # init the trainer and 🚀
    trainer = Trainer(
        TrainerArgs(
            restore_path=CHECKPOINT_PATH,  # Use the specified checkpoint path
            skip_train_epoch=False,
            start_with_eval=START_WITH_EVAL,
            grad_accum_steps=GRAD_ACUMM_STEPS,
        ),
        config,
        output_path=OUT_PATH,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    
    # Enable A100 optimizations
    if torch.cuda.is_available():
        # Enable automatic mixed precision for faster training
        torch.backends.cudnn.benchmark = True  # Enable CUDNN benchmarking for faster training
        # Enable TF32 for better performance on A100 (still maintains good precision)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Only use these if they're supported in your PyTorch version
        try:
            if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
                torch.backends.cuda.enable_mem_efficient_sdp(True)
            if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
                torch.backends.cuda.enable_flash_sdp(True)
        except Exception as e:
            print("Note: Some optimizations not supported in your PyTorch version:", e)
    
    trainer.fit()


if __name__ == "__main__":
    main()