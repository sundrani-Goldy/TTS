import os
import pickle as pickle_tts
from typing import Any, Callable, Dict, Union

import fsspec
import torch

from TTS.utils.generic_utils import get_user_data_dir


class RenamingUnpickler(pickle_tts.Unpickler):
    """Overload default pickler to solve module renaming problem"""

    def find_class(self, module, name):
        return super().find_class(module.replace("mozilla_voice_tts", "TTS"), name)


class AttrDict(dict):
    """A custom dict which converts dict keys
    to class attributes"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def load_fsspec(
    path: str,
    map_location: Union[str, Callable, torch.device, Dict[Union[str, torch.device], Union[str, torch.device]]] = None,
    cache: bool = True,
    **kwargs,
) -> Any:
    """Like torch.load but can load from other locations (e.g. s3:// , gs://).

    Args:
        path: Any path or url supported by fsspec.
        map_location: torch.device or str.
        cache: If True, cache a remote file locally for subsequent calls. It is cached under `get_user_data_dir()/tts_cache`. Defaults to True.
        **kwargs: Keyword arguments forwarded to torch.load.

    Returns:
        Object stored in path.
    """
    #XttsConfig defines all the hyperparameters and settings used by the XTTS model (e.g., model architecture, language support, speaker embedding, audio settings, etc.).
    from TTS.tts.configs.xtts_config import XttsConfig
    #XttsAudioConfig defines the audio configuration settings for the XTTS model (e.g., sample rate, number of mel channels, etc.).
    #XttsArgs defines the arguments used for training and evaluating the XTTS model (e.g., batch size, learning rate, etc.).
    from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
    #BaseDatasetConfig defines the base configuration for datasets used in TTS training and evaluation (e.g., dataset paths, data augmentation settings, etc.).
    #It is used to create a dataset object that can be used for training and evaluation.
    from TTS.config.shared_configs import BaseDatasetConfig
    
    # Add the classes to the safe globals for torch serialization
    # This is necessary to avoid issues with loading the model when the classes are not in the global namespace.
    torch.serialization.add_safe_globals([
    XttsConfig,
    XttsAudioConfig,
    XttsArgs,
    BaseDatasetConfig
])
    is_local = os.path.isdir(path) or os.path.isfile(path)
    if cache and not is_local:
        with fsspec.open(
            f"filecache::{path}",
            filecache={"cache_storage": str(get_user_data_dir("tts_cache"))},
            mode="rb",
        ) as f:
            return torch.load(f, map_location=map_location, **kwargs)
    else:
        with fsspec.open(path, "rb") as f:
            return torch.load(f, map_location=map_location, **kwargs)


def load_checkpoint(
    model, checkpoint_path, use_cuda=False, eval=False, cache=False
):  # pylint: disable=redefined-builtin
    try:
        state = load_fsspec(checkpoint_path, map_location=torch.device("cpu"), cache=cache)
    except ModuleNotFoundError:
        pickle_tts.Unpickler = RenamingUnpickler
        state = load_fsspec(checkpoint_path, map_location=torch.device("cpu"), pickle_module=pickle_tts, cache=cache)
    model.load_state_dict(state["model"])
    if use_cuda:
        model.cuda()
    if eval:
        model.eval()
    return model, state