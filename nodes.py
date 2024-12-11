
import os
import torch
from PIL import Image
import folder_paths

import argparse
import logging
import os
import requests
import subprocess

import torch
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from diffusers.utils.import_utils import is_xformers_available
from packaging import version
from tqdm import tqdm
from funasr.download.download_from_hub import download_model
from funasr.models.emotion2vec.model import Emotion2vec

from .models.wav2vec import Wav2VecModel
from .models.emotion_classifier import AudioEmotionClassifierModel
from .models.unet_2d_condition import UNet2DConditionModel
from .models.unet_3d import UNet3DConditionModel
from .models.audio_proj import AudioProjModel
from .models.image_proj import ImageProjModel
from .pipelines.video_pipeline import VideoPipeline

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from torchvision import transforms
from transformers import Wav2Vec2FeatureExtractor
from audio_separator.separator import Separator
from io import BytesIO
import librosa
import math
from einops import rearrange
import torchaudio
import torch.nn.functional as F
from diffusers.utils.import_utils import is_xformers_available

from moviepy.editor import AudioFileClip, VideoClip


def tensor2pil(t_image: torch.Tensor)  -> Image:
    return Image.fromarray(np.clip(255.0 * t_image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image:Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

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

# 下载hg 模型到本地
def download_hg_model(model_id:str,exDir:str=''):
    # 下载本地
    model_checkpoint = os.path.join(folder_paths.models_dir, exDir, os.path.basename(model_id))
    print(model_checkpoint)
    if not os.path.exists(model_checkpoint):
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=model_id, local_dir=model_checkpoint, local_dir_use_symlinks=False)
    return model_checkpoint

def tensor_to_video(tensor, output_video_path, input_audio_path, fps=30, max_duration=None):
    """
    Converts a Tensor with shape [c, f, h, w] into a video and adds an audio track.
    
    Args:
        tensor (Tensor): The Tensor to be converted, shaped [c, f, h, w].
        output_video_path (str): The file path where the output video will be saved.
        input_audio_path (str): The path to the audio file (WAV file).
        fps (int): The frame rate of the output video. Default is 30 fps.
        max_duration (float, optional): Maximum duration of the video in seconds.
    """
    tensor = tensor.permute(1, 2, 3, 0).cpu().numpy()  # convert to [f, h, w, c]
    tensor = np.clip(tensor * 255, 0, 255).astype(np.uint8)  # to [0, 255]
    
    video_duration = tensor.shape[0] / fps
    
    # Optional: Limit duration if max_duration is specified
    if max_duration is not None:
        video_duration = min(video_duration, max_duration)
        tensor = tensor[:int(max_duration * fps)]
    
    def make_frame(t):
        frame_index = min(int(t * fps), tensor.shape[0] - 1)
        return tensor[frame_index]
    
    audio_clip = AudioFileClip(input_audio_path)
    audio_duration = audio_clip.duration
    final_duration = min(video_duration, audio_duration)
    
    audio_clip = audio_clip.subclip(0, final_duration)
    new_video_clip = VideoClip(make_frame, duration=final_duration)
    new_video_clip = new_video_clip.set_audio(audio_clip)
    new_video_clip.write_videofile(output_video_path, fps=fps, audio_codec="aac")

def tensor_to_images(tensor, output_dir, prefix='frame', format='png', max_duration=None, fps=30):
    """
    Converts a Tensor with shape [c, f, h, w] into individual image files.
    
    Args:
        tensor (Tensor): The Tensor to be converted, shaped [c, f, h, w].
        output_dir (str): The directory where image frames will be saved.
        prefix (str): Prefix for the image filenames. Default is 'frame'.
        format (str): Image file format. Default is 'png'.
        max_duration (float, optional): Maximum duration in seconds.
        fps (int, optional): Frames per second. Default is 30.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert tensor to [f, h, w, c] format
    tensor = tensor.permute(1, 2, 3, 0).cpu().numpy()
    
    # Optional: Limit number of frames if max_duration is specified
    if max_duration is not None:
        max_frames = int(max_duration * fps)
        tensor = tensor[:max_frames]
    
    # Clip and convert to uint8
    tensor = np.clip(tensor * 255, 0, 255).astype(np.uint8)
    
    # Save each frame as an image
    for i, frame in enumerate(tensor):
        # Create filename with zero-padded index
        filename = os.path.join(output_dir, f"{prefix}_{i:04d}.{format}")
        
        # Save the frame
        Image.fromarray(frame).save(filename)
    
    print(f"Saved {len(tensor)} frames to {output_dir}")
    return output_dir

def images_to_video(image_paths, output_video_path, input_audio_path, fps=30):
    """
    Converts a list of image paths into a video and adds an audio track.
    
    Args:
        image_paths (list): List of file paths to image frames.
        output_video_path (str): The file path where the output video will be saved.
        input_audio_path (str): The path to the audio file (WAV file).
        fps (int): The frame rate of the output video. Default is 30 fps.
    """
    # Read first image to get dimensions
    first_image = Image.open(image_paths[0])
    width, height = first_image.size
    
    # Create video clip from images
    def make_frame(t):
        frame_index = min(int(t * fps), len(image_paths) - 1)
        return np.array(Image.open(image_paths[frame_index]))
    
    # Calculate video duration
    video_duration = len(image_paths) / fps
    
    # Handle audio
    audio_clip = AudioFileClip(input_audio_path)
    audio_duration = audio_clip.duration
    final_duration = min(video_duration, audio_duration)
    
    audio_clip = audio_clip.subclip(0, final_duration)
    new_video_clip = VideoClip(make_frame, duration=final_duration)
    new_video_clip = new_video_clip.set_audio(audio_clip)
    
    # Write video file
    new_video_clip.write_videofile(output_video_path, fps=fps, audio_codec="aac")

# def tensor_to_video(tensor, output_video_path, input_audio_path, fps=30):
#     """
#     Converts a Tensor with shape [c, f, h, w] into a video and adds an audio track from the specified audio file.

#     Args:
#         tensor (Tensor): The Tensor to be converted, shaped [c, f, h, w].
#         output_video_path (str): The file path where the output video will be saved.
#         input_audio_path (str): The path to the audio file (WAV file) that contains the audio track to be added.
#         fps (int): The frame rate of the output video. Default is 30 fps.
#     """
#     tensor = tensor.permute(1, 2, 3, 0).cpu().numpy()  # convert to [f, h, w, c]
#     tensor = np.clip(tensor * 255, 0, 255).astype(np.uint8)  # to [0, 255]

#     def make_frame(t):
#         frame_index = min(int(t * fps), tensor.shape[0] - 1)
#         return tensor[frame_index]

#     video_duration = tensor.shape[0] / fps
#     audio_clip = AudioFileClip(input_audio_path)
#     audio_duration = audio_clip.duration
#     final_duration = min(video_duration, audio_duration)
#     audio_clip = audio_clip.subclip(0, final_duration)
#     new_video_clip = VideoClip(make_frame, duration=final_duration)
#     new_video_clip = new_video_clip.set_audio(audio_clip)
#     new_video_clip.write_videofile(output_video_path, fps=fps, audio_codec="aac")



# def tensor_to_images(tensor, output_dir, prefix='frame', format='png'):
#     """
#     Converts a Tensor with shape [c, f, h, w] into individual image files.
    
#     Args:
#         tensor (Tensor): The Tensor to be converted, shaped [c, f, h, w].
#         output_dir (str): The directory where image frames will be saved.
#         prefix (str): Prefix for the image filenames. Default is 'frame'.
#         format (str): Image file format. Default is 'png'.
#     """
#     # Ensure output directory exists
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Convert tensor to [f, h, w, c] format
#     tensor = tensor.permute(1, 2, 3, 0).cpu().numpy()
    
#     # Clip and convert to uint8
#     tensor = np.clip(tensor * 255, 0, 255).astype(np.uint8)
    
#     # Save each frame as an image
#     for i, frame in enumerate(tensor):
#         # Create filename with zero-padded index
#         filename = os.path.join(output_dir, f"{prefix}_{i:04d}.{format}")
        
#         # Save the frame
#         Image.fromarray(frame).save(filename)
    
#     print(f"Saved {len(tensor)} frames to {output_dir}")
#     return output_dir

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
    print(f"Emotion2vec checkpoint: {path}")
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
                        print(f"init param, map: {k} from {k_src} in ckpt")
                elif k.startswith(dst_prefix) and k.replace(dst_prefix, src_prefix, 1) in src_state.keys():
                    k_src = k.replace(dst_prefix, src_prefix, 1)
                    if not k_src.startswith("module."):
                        print(f"init param, map: {k} from {k_src} in ckpt")

        if k_src in src_state.keys():
            if ignore_init_mismatch and dst_state[k].shape != src_state[k_src].shape:
                print(
                    f"ignore_init_mismatch:{ignore_init_mismatch}, dst: {k, dst_state[k].shape}, src: {k_src, src_state[k_src].shape}"
                )
            else:
                dst_state[k] = src_state[k_src]

        else:
            print(f"Warning, miss key in ckpt: {k}, mapped: {k_src}")

    obj.load_state_dict(dst_state, strict=True)
    

class CXH_Memo_Load:

    def __init__(self):
        self.model = None
        self.weight_dtype = None
        self.SetNone()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (["memoavatar/memo"],),
                "weight_dtype": (["fp16","bf16","fp32"],),
            }
        }

    RETURN_TYPES = ("CXH_Memo_Load",) #RETURN_TYPES = () RETURN_TYPES = ("DICT",)返回字典
    RETURN_NAMES = ("pipe",)
    FUNCTION = "gen"
    OUTPUT_NODE = False #OUTPUT_NODE = True 没输出
    CATEGORY = "CXH/example"

    def gen(self, model,weight_dtype): 
        self.model = model
        self.weight_dtype = weight_dtype
        if  self.face_analysis == None:
            self.run()
        return (self,)
        
    def SetNone(self):
        self.face_analysis = None
        self.wav2vec_feature_extractor = None
        self.vocal_separator =None
        self.emotion_model = None
        self.classifier = None
        self.pipeline = None
        self.device = None
        self.audio_encoder = None
        self.reference_net = None
        self.diffusion_net = None
        self.audio_proj = None

    def clear(self):
        del self.face_analysis
        del self.emotion_model
        del self.classifier
        self.SetNone()


    def run(self):
        model = self.model
        weight_dtype = self.weight_dtype

        memo_dir = download_hg_model(model,"memo")

        input_dir =  os.path.join(folder_paths.models_dir,"memo")

        vocal_separator = os.path.join(memo_dir, "misc/vocal_separator/Kim_Vocal_2.onnx")
        
        if os.path.exists(vocal_separator):
            print("need download model : memoavatar/memo move to models/memo")

        # Set up device and weight dtype
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        self.device = device
        if weight_dtype == "fp16":
            weight_dtype = torch.float16
        elif weight_dtype == "bf16":
            weight_dtype = torch.bfloat16
        elif weight_dtype == "fp32":
            weight_dtype = torch.float32
        else:
            weight_dtype = torch.float32
        print(f"Inference dtype: {weight_dtype}")

        # Initialize the FaceAnalysis model
        face_analysis_model = os.path.join(memo_dir, "misc/face_analysis")
        face_analysis = FaceAnalysis(
            name="",
            root=face_analysis_model,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        face_analysis.prepare(ctx_id=0, det_size=(640, 640))
       
        
         # Initialize Wav2Vec model
        # wav2vec_model =  os.path.join(input_dir, "wav2vec2-base-960h")
        wav2vec_model = download_hg_model("facebook/wav2vec2-base-960h","memo")
        audio_encoder = Wav2VecModel.from_pretrained(wav2vec_model).to(device=device)
        audio_encoder.feature_extractor._freeze_parameters()
        # Initialize Wav2Vec feature extractor
        wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec_model)

        vocal_separator_model = vocal_separator = os.path.join(memo_dir, "misc/vocal_separator/Kim_Vocal_2.onnx")
        cache_dir = os.path.join(input_dir, "cache_dir")
        audio_preprocess_dir = os.path.join(cache_dir, "audio_preprocess")
        vocal_separator = None
        if vocal_separator_model is not None:
            os.makedirs(cache_dir, exist_ok=True)
            vocal_separator = Separator(
                output_dir=audio_preprocess_dir,
                output_single_stem="vocals",
                model_file_dir=os.path.dirname(vocal_separator_model),
            )
            vocal_separator.load_model(os.path.basename(vocal_separator_model))
            assert vocal_separator.model_instance is not None, "Failed to load audio separation model."

        # Load models
        print("Downloading emotion2vec models from modelscope")
        emotion2vec_model = wav2vec_model =  os.path.join(input_dir, "emotion2vec_plus_large")
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

        vae_path = download_hg_model("stabilityai/sd-vae-ft-mse","memo")
        # vae_path = wav2vec_model =  os.path.join(input_dir, "sd-vae-ft-mse")
        vae = AutoencoderKL.from_pretrained(vae_path).to(device=device, dtype=weight_dtype)
        
        model_name_or_path = memo_dir
        reference_net = UNet2DConditionModel.from_pretrained(
            model_name_or_path, subfolder="reference_net", use_safetensors=True
        )
        diffusion_net = UNet3DConditionModel.from_pretrained(
            model_name_or_path, subfolder="diffusion_net", use_safetensors=True
        )
        image_proj = ImageProjModel.from_pretrained(
            model_name_or_path, subfolder="image_proj", use_safetensors=True
        )
        audio_proj = AudioProjModel.from_pretrained(
            model_name_or_path, subfolder="audio_proj", use_safetensors=True
        )

        vae.requires_grad_(False).eval()
        reference_net.requires_grad_(False).eval()
        diffusion_net.requires_grad_(False).eval()
        image_proj.requires_grad_(False).eval()
        audio_proj.requires_grad_(False).eval()

        noise_scheduler = FlowMatchEulerDiscreteScheduler()
        pipeline = VideoPipeline(
            vae=vae,
            reference_net=reference_net,
            diffusion_net=diffusion_net,
            scheduler=noise_scheduler,
            image_proj=image_proj,
        )
        pipeline.to(device=device, dtype=weight_dtype)

        self.face_analysis = face_analysis
        self.wav2vec_feature_extractor = wav2vec_feature_extractor
        self.vocal_separator = vocal_separator
        self.emotion_model = emotion_model
        self.classifier = classifier
        self.pipeline = pipeline
        self.audio_encoder = audio_encoder
        self.audio_proj = audio_proj
        self.diffusion_net = diffusion_net
        self.reference_net =reference_net

# class CXH_Memo_Restore:
#     def __init__(self):
#         pass

#     @classmethod
#     def INPUT_TYPES(s):
#         return {
#             "required": {
#                "frame_path":("STRING", {"multiline": True, "default": "", "forceInput": True},),
#                "audio_path":("STRING", {"multiline": True, "default": "", "forceInput": True},),
#                "out_path":("STRING", {"multiline": True, "default": "",},),
#                "fps":("INT", {"default": 30, "min": 1, "max": 120}),
#             }
#         }

#     RETURN_TYPES = () #RETURN_TYPES = () RETURN_TYPES = ("DICT",)返回字典
#     RETURN_NAMES = ()
#     FUNCTION = "gen"
#     OUTPUT_NODE = True #OUTPUT_NODE = True 没输出
#     CATEGORY = "CXH/example"

#     def gen(self,frame_path,audio_path,out_path,fps): 
#         images_to_video(image_paths = frame_path, output_video_path = out_path, input_audio_path = audio_path, fps=fps)
#         return()

class CXH_Memo_Run:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe": ("CXH_Memo_Load",),
                "image": ("IMAGE",),
                "audio_path":   ("STRING", {"multiline": False, "default": ""},),
                "output_dir":   ("STRING", {"multiline": False, "default": ""},),
                "fps":("INT", {"default": 30, "min": 1, "max": 120}),
                "frames_per_clip":("INT", {"default": 16, "min": 1, "max": 64}),
                "steps":("INT", {"default": 20, "min": 1, "max": 120}),
                "cfg":("FLOAT", {"default": 3.5, "min": 1, "max": 40}),
                "width":("INT", {"default": 512, "min": 1, "max": 1024}),
                "height":("INT", {"default": 512, "min": 1, "max": 1024}),
                "cache": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 656545, "min": 0, "max": 1000000}),
            },
            
        }

    RETURN_TYPES = ("STRING","STRING",) #RETURN_TYPES = () RETURN_TYPES = ("DICT",)返回字典
    RETURN_NAMES = ("video_path","frame_path",)
    FUNCTION = "gen"
    OUTPUT_NODE = False #OUTPUT_NODE = True 没输出
    CATEGORY = "CXH/example"

    def gen(self, pipe,image,audio_path,output_dir,fps,frames_per_clip,steps,cfg,width,height,cache,seed):
        if "wav" not in audio_path:
            print("MEMO might not generate full-length video for non-wav audio file.")
        input_dir =  os.path.join(folder_paths.models_dir,"memo")
        cache_dir = output_dir #os.path.join(input_dir, "cache_dir")
        os.makedirs(cache_dir, exist_ok=True)

        if pipe.face_analysis == None:
            pipe.run()

        face_analysis = pipe.face_analysis
        wav2vec_feature_extractor = pipe.wav2vec_feature_extractor
        vocal_separator = pipe.vocal_separator 
        emotion_model = pipe.emotion_model
        classifier = pipe.classifier
        pipeline = pipe.pipeline
        device = pipe.device
        audio_encoder = pipe.audio_encoder
        reference_net = pipe.reference_net
        diffusion_net = pipe.diffusion_net
        audio_proj = pipe.audio_proj

        sample_rate = 16000
        # fps = 30
        num_generated_frames_per_clip = frames_per_clip  #控制视频生成的帧片段长度原代码中设置为16帧
        # size = 512
        num_init_past_frames = 2
        num_past_frames = 16
        inference_steps= steps
        cfg_scale = cfg

        # 图片处理
        image = tensor2pil(image)
        transform = transforms.Compose(
        [
            transforms.Resize((width, height)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
            ]
        )
        pixel_values = transform(image)
        pixel_values = pixel_values.unsqueeze(0)
        image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        faces = face_analysis.get(image_bgr)
        if not faces:
            print("No faces detected in the image. Using a zero vector as the face embedding.")
            face_emb = np.zeros(512)
        else:
            # Sort faces by size and select the largest one
            faces_sorted = sorted(
                faces,
                key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]),
                reverse=True,
            )
            face_emb = faces_sorted[0]["embedding"]
        face_emb = face_emb.reshape(1, -1)
        face_emb = torch.tensor(face_emb)

       
        audio_preprocess_dir = os.path.join(cache_dir, "audio_preprocess")
        os.makedirs(audio_preprocess_dir, exist_ok=True)
        
        audio_path = resample_audio(
            audio_path,
            os.path.join(audio_preprocess_dir, f"{os.path.basename(audio_path).split('.')[0]}-16k.wav"),
        )

        if vocal_separator is not None:
            outputs = vocal_separator.separate(audio_path)
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
            vocal_audio_file = audio_path

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

        wav, sr = torchaudio.load(audio_path)
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

        if is_xformers_available():
            import xformers
            xformers_version = version.parse(xformers.__version__)
            print("xformers version :" + str(xformers_version))
            
            # 添加非空检查
            if reference_net is not None:
                reference_net.enable_xformers_memory_efficient_attention()

            if diffusion_net is not None:
                diffusion_net.enable_xformers_memory_efficient_attention()
        else:
            print("xformers is not available. Proceeding without memory efficient attention.")
        video_frames = []
        num_clips = audio_emb.shape[0] // num_generated_frames_per_clip
        generator=torch.manual_seed(seed)
        
        for t in tqdm(range(num_clips), desc="Generating video clips"):
            if len(video_frames) == 0:
                # Initialize the first past frames with reference image
                past_frames = pixel_values.repeat(num_init_past_frames, 1, 1, 1)
                past_frames = past_frames.to(dtype=pixel_values.dtype, device=pixel_values.device)
                pixel_values_ref_img = torch.cat([pixel_values, past_frames], dim=0)
            else:
                past_frames = video_frames[-1][0]
                past_frames = past_frames.permute(1, 0, 2, 3)
                past_frames = past_frames[0 - num_past_frames :]
                past_frames = past_frames * 2.0 - 1.0
                past_frames = past_frames.to(dtype=pixel_values.dtype, device=pixel_values.device)
                pixel_values_ref_img = torch.cat([pixel_values, past_frames], dim=0)

            pixel_values_ref_img = pixel_values_ref_img.unsqueeze(0)

            audio_tensor = (
                audio_emb[
                    t
                    * num_generated_frames_per_clip : min(
                        (t + 1) * num_generated_frames_per_clip, audio_emb.shape[0]
                    )
                ]
                .unsqueeze(0)
                .to(device= audio_proj.device, dtype=audio_proj.dtype)
            )
            audio_tensor = audio_proj(audio_tensor)

            audio_emotion = emotion_labels
            audio_emotion_tensor = audio_emotion[
                t
                * num_generated_frames_per_clip : min(
                    (t + 1) * num_generated_frames_per_clip, audio_emb.shape[0]
                )
            ]

            pipeline_output = pipeline(
                ref_image=pixel_values_ref_img,
                audio_tensor=audio_tensor,
                audio_emotion=audio_emotion_tensor,
                emotion_class_num=num_emotion_classes,
                face_emb=face_emb,
                width=width,
                height=height,
                video_length=num_generated_frames_per_clip,
                num_inference_steps=inference_steps,
                guidance_scale=cfg_scale,
                generator=generator,
            )

            video_frames.append(pipeline_output.videos)

        video_frames = torch.cat(video_frames, dim=2)
        video_frames = video_frames.squeeze(0)
        video_frames = video_frames[:, :audio_length]

        output_dir = os.path.join(cache_dir,"out")
        os.makedirs(output_dir, exist_ok=True)
        output_video_path = os.path.join(
                output_dir,
                f"{seed}.mp4",
            )

        tensor_to_video(video_frames, output_video_path, audio_path, fps=fps)

        framePath = os.path.join(output_dir,str(seed))
        tensor_to_images(tensor=video_frames,output_dir=framePath,fps=30)
            
        # results = []
        # if frame == True:
        #     # Convert tensor to [f, h, w, c] format
        #     tensor = video_frames.permute(1, 2, 3, 0).cpu().numpy()
            
        #     # Clip and convert to uint8
        #     tensor = np.clip(tensor * 255, 0, 255).astype(np.uint8)
           
        #     # Save each frame as an image
        #     for i, frame in enumerate(tensor):
            
        #         pil = Image.fromarray(frame)
        #         results.append(pil2tensor(pil))

        if cache == False:
            pipe.clear()


        return(output_video_path,framePath,)