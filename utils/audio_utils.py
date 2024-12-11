import logging
import math
import os
import subprocess
from io import BytesIO

import librosa
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from audio_separator.separator import Separator
from einops import rearrange
from funasr.download.download_from_hub import download_model
from funasr.models.emotion2vec.model import Emotion2vec
from transformers import Wav2Vec2FeatureExtractor

from memo.models.emotion_classifier import AudioEmotionClassifierModel
from memo.models.wav2vec import Wav2VecModel


logger = logging.getLogger(__name__)


def resample_audio(input_audio_file: str, output_audio_file: str, sample_rate: int = 16000):
    p = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            input_audio_file,
            "-ar",
            str(sample_rate),
            output_audio_file,
        ]
    )
    ret = p.wait()
    assert ret == 0, f"Resample audio failed! Input: {input_audio_file}, Output: {output_audio_file}"
    return output_audio_file


@torch.no_grad()
def preprocess_audio(
    wav_path: str,
    fps: int,
    wav2vec_model: str,
    vocal_separator_model: str = None,
    cache_dir: str = "",
    device: str = "cuda",
    sample_rate: int = 16000,
    num_generated_frames_per_clip: int = -1,
):
    """
    Preprocess the audio file and extract audio embeddings.

    Args:
        wav_path (str): Path to the input audio file.
        fps (int): Frames per second for the audio processing.
        wav2vec_model (str): Path to the pretrained Wav2Vec model.
        vocal_separator_model (str, optional): Path to the vocal separator model. Defaults to None.
        cache_dir (str, optional): Directory for cached files. Defaults to "".
        device (str, optional): Device to use ('cuda' or 'cpu'). Defaults to "cuda".
        sample_rate (int, optional): Sampling rate for audio processing. Defaults to 16000.
        num_generated_frames_per_clip (int, optional): Number of generated frames per clip for padding. Defaults to -1.

    Returns:
        tuple: A tuple containing:
            - audio_emb (torch.Tensor): The processed audio embeddings.
            - audio_length (int): The length of the audio in frames.
    """
    # Initialize Wav2Vec model
    audio_encoder = Wav2VecModel.from_pretrained(wav2vec_model).to(device=device)
    audio_encoder.feature_extractor._freeze_parameters()

    # Initialize Wav2Vec feature extractor
    wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec_model)

    # Initialize vocal separator if provided
    vocal_separator = None
    if vocal_separator_model is not None:
        os.makedirs(cache_dir, exist_ok=True)
        vocal_separator = Separator(
            output_dir=cache_dir,
            output_single_stem="vocals",
            model_file_dir=os.path.dirname(vocal_separator_model),
        )
        vocal_separator.load_model(os.path.basename(vocal_separator_model))
        assert vocal_separator.model_instance is not None, "Failed to load audio separation model."

    # Perform vocal separation if applicable
    if vocal_separator is not None:
        outputs = vocal_separator.separate(wav_path)
        assert len(outputs) > 0, "Audio separation failed."
        vocal_audio_file = outputs[0]
        vocal_audio_name, _ = os.path.splitext(vocal_audio_file)
        vocal_audio_file = os.path.join(vocal_separator.output_dir, vocal_audio_file)
        vocal_audio_file = resample_audio(
            vocal_audio_file,
            os.path.join(vocal_separator.output_dir, f"{vocal_audio_name}-16k.wav"),
            sample_rate,
        )
    else:
        vocal_audio_file = wav_path

    # Load audio and extract Wav2Vec features
    speech_array, sampling_rate = librosa.load(vocal_audio_file, sr=sample_rate)
    audio_feature = np.squeeze(wav2vec_feature_extractor(speech_array, sampling_rate=sampling_rate).input_values)
    audio_length = math.ceil(len(audio_feature) / sample_rate * fps)
    audio_feature = torch.from_numpy(audio_feature).float().to(device=device)

    # Pad audio features to match the required length
    if num_generated_frames_per_clip > 0 and audio_length % num_generated_frames_per_clip != 0:
        audio_feature = torch.nn.functional.pad(
            audio_feature,
            (
                0,
                (num_generated_frames_per_clip - audio_length % num_generated_frames_per_clip) * (sample_rate // fps),
            ),
            "constant",
            0.0,
        )
        audio_length += num_generated_frames_per_clip - audio_length % num_generated_frames_per_clip
    audio_feature = audio_feature.unsqueeze(0)

    # Extract audio embeddings
    with torch.no_grad():
        embeddings = audio_encoder(audio_feature, seq_len=audio_length, output_hidden_states=True)
    assert len(embeddings) > 0, "Failed to extract audio embeddings."
    audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
    audio_emb = rearrange(audio_emb, "b s d -> s b d")

    # Concatenate embeddings with surrounding frames
    audio_emb = audio_emb.cpu().detach()
    concatenated_tensors = []
    for i in range(audio_emb.shape[0]):
        vectors_to_concat = [audio_emb[max(min(i + j, audio_emb.shape[0] - 1), 0)] for j in range(-2, 3)]
        concatenated_tensors.append(torch.stack(vectors_to_concat, dim=0))
    audio_emb = torch.stack(concatenated_tensors, dim=0)

    if vocal_separator is not None:
        del vocal_separator
    del audio_encoder

    return audio_emb, audio_length


@torch.no_grad()
def extract_audio_emotion_labels(
    model: str,
    wav_path: str,
    emotion2vec_model: str,
    audio_length: int,
    sample_rate: int = 16000,
    device: str = "cuda",
):
    """
    Extract audio emotion labels from an audio file.

    Args:
        model (str): Path to the MEMO model.
        wav_path (str): Path to the input audio file.
        emotion2vec_model (str): Path to the Emotion2vec model.
        audio_length (int): Target length for interpolated emotion labels.
        sample_rate (int, optional): Sample rate of the input audio. Default is 16000.
        device (str, optional): Device to use ('cuda' or 'cpu'). Default is "cuda".

    Returns:
        torch.Tensor: Processed emotion labels with shape matching the target audio length.
    """
    # Load models
    logger.info("Downloading emotion2vec models from modelscope")
    kwargs = download_model(model=emotion2vec_model)
    kwargs["tokenizer"] = None
    kwargs["input_size"] = None
    kwargs["frontend"] = None
    emotion_model = Emotion2vec(**kwargs, vocab_size=-1).to(device)
    init_param = kwargs.get("init_param", None)
    load_emotion2vec_model(
        model=emotion_model,
        path=init_param,
        ignore_init_mismatch=kwargs.get("ignore_init_mismatch", True),
        oss_bucket=kwargs.get("oss_bucket", None),
        scope_map=kwargs.get("scope_map", []),
    )
    emotion_model.eval()

    classifier = AudioEmotionClassifierModel.from_pretrained(
        model,
        subfolder="misc/audio_emotion_classifier",
        use_safetensors=True,
    ).to(device=device)
    classifier.eval()

    # Load audio
    wav, sr = torchaudio.load(wav_path)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    wav = wav.view(-1) if wav.dim() == 1 else wav[0].view(-1)

    emotion_labels = torch.full_like(wav, -1, dtype=torch.int32)

    def extract_emotion(x):
        """
        Extract emotion for a given audio segment.
        """
        x = x.to(device=device)
        x = F.layer_norm(x, x.shape).view(1, -1)
        feats = emotion_model.extract_features(x)
        x = feats["x"].mean(dim=1)  # average across frames
        x = classifier(x)
        x = torch.softmax(x, dim=-1)
        return torch.argmax(x, dim=-1)

    # Process start, middle, and end segments
    start_label = extract_emotion(wav[: sample_rate * 2]).item()
    emotion_labels[:sample_rate] = start_label

    for i in range(sample_rate, len(wav) - sample_rate, sample_rate):
        mid_wav = wav[i - sample_rate : i - sample_rate + sample_rate * 3]
        mid_label = extract_emotion(mid_wav).item()
        emotion_labels[i : i + sample_rate] = mid_label

    end_label = extract_emotion(wav[-sample_rate * 2 :]).item()
    emotion_labels[-sample_rate:] = end_label

    # Interpolate to match the target audio length
    emotion_labels = emotion_labels.unsqueeze(0).unsqueeze(0).float()
    emotion_labels = F.interpolate(emotion_labels, size=audio_length, mode="nearest").squeeze(0).squeeze(0).int()
    num_emotion_classes = classifier.num_emotion_classes

    del emotion_model
    del classifier

    return emotion_labels, num_emotion_classes


def load_emotion2vec_model(
    path: str,
    model: torch.nn.Module,
    ignore_init_mismatch: bool = True,
    map_location: str = "cpu",
    oss_bucket=None,
    scope_map=[],
):
    obj = model
    dst_state = obj.state_dict()
    logger.debug(f"Emotion2vec checkpoint: {path}")
    if oss_bucket is None:
        src_state = torch.load(path, map_location=map_location)
    else:
        buffer = BytesIO(oss_bucket.get_object(path).read())
        src_state = torch.load(buffer, map_location=map_location)

    src_state = src_state["state_dict"] if "state_dict" in src_state else src_state
    src_state = src_state["model_state_dict"] if "model_state_dict" in src_state else src_state
    src_state = src_state["model"] if "model" in src_state else src_state

    if isinstance(scope_map, str):
        scope_map = scope_map.split(",")
    scope_map += ["module.", "None"]

    for k in dst_state.keys():
        k_src = k
        if scope_map is not None:
            src_prefix = ""
            dst_prefix = ""
            for i in range(0, len(scope_map), 2):
                src_prefix = scope_map[i] if scope_map[i].lower() != "none" else ""
                dst_prefix = scope_map[i + 1] if scope_map[i + 1].lower() != "none" else ""

                if dst_prefix == "" and (src_prefix + k) in src_state.keys():
                    k_src = src_prefix + k
                    if not k_src.startswith("module."):
                        logger.debug(f"init param, map: {k} from {k_src} in ckpt")
                elif k.startswith(dst_prefix) and k.replace(dst_prefix, src_prefix, 1) in src_state.keys():
                    k_src = k.replace(dst_prefix, src_prefix, 1)
                    if not k_src.startswith("module."):
                        logger.debug(f"init param, map: {k} from {k_src} in ckpt")

        if k_src in src_state.keys():
            if ignore_init_mismatch and dst_state[k].shape != src_state[k_src].shape:
                logger.debug(
                    f"ignore_init_mismatch:{ignore_init_mismatch}, dst: {k, dst_state[k].shape}, src: {k_src, src_state[k_src].shape}"
                )
            else:
                dst_state[k] = src_state[k_src]

        else:
            logger.debug(f"Warning, miss key in ckpt: {k}, mapped: {k_src}")

    obj.load_state_dict(dst_state, strict=True)
