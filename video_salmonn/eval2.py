import os
from config.config import Config
import argparse
import yaml
import json
from omegaconf import OmegaConf
import tempfile
import traceback
tempfile.tempdir = "/share/nlp/tuwenming/projects/HAVIB/tmp"

from datasets import SupervisedAudioVisualDataset4Test
from model.openllama import OpenLLAMAPEFTModel
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

from typing import List, Optional
from moviepy.editor import (
    AudioFileClip,
    concatenate_audioclips,
    ImageClip,
    AudioClip,
    concatenate_videoclips,
    VideoFileClip
)
# 本文件用于保存所有的定义
maic_cls_list = ['bus', 'hair-dryer', 'pipa', 'man', 'ambulance', 'razor', 'harp', 'tabla', 'bass', 'handpan', 
        'girl', 'sitar', 'car', 'lion', 'guitar', 'vacuum-cleaner', 'cat', 'mower', 'helicopter', 'boy', 'drum', 
        'keyboard', 'tuba', 'saw', 'flute', 'cello', 'woman', 'gun', 'accordion', 'violin', 'clarinet', 'erhu', 
        'saxophone', 'guzheng', 'dog', 'baby', 'horse', 'male', 'wolf', 'bird', 'ukulele', 'piano', 'female', 
        'marimba', 'not sure', 'no available option']

mvic_cls_list = ['sushi', 'banana', 'cake', 'butterfly', 'bird', 'microphone', 'hamburger', 'pineapple', 
        'man', 'book', 'sunglasses', 'goat', 'tie', 'cabinetry', 'motorcycle', 'drawer', 'strawberry', 
        'sheep', 'pasta', 'parrot', 'bull', 'table', 'penguin', 'watch', 'pillow', 'shellfish', 'kangaroo', 
        'flower', 'paddle', 'rocket', 'helicopter', 'bus', 'mushroom', 'bee', 'tree', 'boat', 'saxophone', 
        'football', 'lizard', 'violin', 'dog', 'cucumber', 'cello', 'airplane', 'horse', 'drum', 'box', 
        'rabbit', 'car', 'door', 'orange', 'shelf', 'camera', 'poster', 'lemon', 'cat', 'fish', 'bread', 
        'piano', 'apple', 'glasses', 'bicycle', 'truck', 'deer', 'woman', 'wheelchair', 'cheese', 'chair', 
        'plate', 'tomato', 'bed', 'starfish', 'balloon', 'bottle', 'crab', 'beer', 'frog', 'shrimp', 'tower', 
        'guitar', 'pig', 'peach', 'train', 'pumpkin', 'elephant', 'jellyfish', 'parachute', 'monkey', 'flag',
        'not sure', 'no available option']


pmp_avl_ans_format = "answer={'category1_id1': '[x_min, y_min, x_max, y_max]', 'category2_id2': '[x_min, y_min, x_max, y_max]'}"
avl_cls_list = ['dog', 'clarinet', 'banjo', 'cat', 'guzheng', 'tree', 'lion', 'tuba', 
        'ukulele', 'flute', 'piano', 'person', 'violin', 'airplane', 'bass', 'pipa', 
        'trumpet', 'accordion', 'saxophone', 'car', 'lawn-mower', 'cello', 'bassoon', 
        'horse', 'guitar', 'erhu', 'not sure', 'no available option']
prompt_avl = f"""
        ctaegories list: {avl_cls_list}
        (1) There may be multiple sounding instances, you can choose instance categories from the given categories list.
        (2) The naming format for instances is: category_id. Instance IDs start from 1, e.g., male_1, dog_2, dog_3, cat_4. 
        (3) The bbox format is: [x_min, y_min, x_max, y_max], where x_min, y_min represent the coordinates of the top-left corner. 
        (4) The bbox values should be normalized into the range of 0 and 1, e.g., [0.1, 0.12, 0.26, 0.14].
        Do not explain, you must strictly follow the format: {pmp_avl_ans_format}
    """

prompt_avlg = """
        Please output the answer in a format that strictly matches the following example, do not explain:
        answer={'frame_0': [x0_min, y0_min, x0_max, y0_max], 'frame_1': None, ..., 'frame_9': [x9, y9, w9, h9]}.
        Note, 
        (1) x_min, y_min represent the coordinates of the top-left corner, while x_max, y_max for the bottom_right corner.
        (2) The bbox values should be normalized into the range of 0 and 1, e.g., [0.1, 0.12, 0.26, 0.14]. 
        (3) Frames should be ranged from frame_0 to frame_9.
    """

avqa_cls_list = ['ukulele', 'cello', 'clarinet', 'violin', 'bassoon', 'accordion', 'banjo', 'tuba', 'flute', 'electric_bass', 'bagpipe', 
        'drum', 'congas', 'suona', 'xylophone', 'saxophone', 'guzheng', 'trumpet', 'erhu', 'piano', 'acoustic_guitar', 'pipa', 'not sure', 'no available option']

havib_constants = {
    'L1_LAQA': {
        'options_sound_clarity': ['first', 'last', 'same', 'not sure'],
        'options_sound_order': ['sound', 'noise', 'not sure'],
        'options_sound_volume': ['first', 'last', 'same', 'not sure'],
        'options_yes_no': ['yes', 'no', 'not sure'],
    },

    'L1_LIQA': {
        'get_from_background_binary': ['yes', 'no', 'not sure'],
        'get_from_image_binary': ['yes', 'no', 'not sure'],
        'get_from_foreground_binary': ['yes', 'no', 'not sure'],
        'get_from_image_triple': ['blurred', 'normal', 'clear', 'not sure'],
        'get_from_3d-task1': ['center', 'left', 'right', 'not sure'],
        'get_from_3d-task2': ['cone', 'cube', 'cylinder', 'cuboid', 'no available option', 'not sure'],
        # 'get_from_3d-task3': [0, 1, 2, 3, 4, 5, 6],
        'get_from_space_hard': ['center', 'top left', 'top center', 'top right', 'bottom left', 'bottom center', 'bottom right', 'no available option', 'not sure'],
        'get_from_color': ['blue', 'green', 'red', 'puprle', 'yellow', 'no available option', 'not sure'],
        'get_yes_no': ['yes', 'no', 'not sure'],
        # 'get_lines_count': [0, 1, 2, 3, 4],
        'get_lines_direction': ['horizontal', 'vertical', 'inclined', 'not sure'],
        'get_from_space_easy_area': ['the right one', 'the left one', 'the middle one', 'the bottom one', 'the top one'],
        'get_from_space_easy_bbrightness': ['the right one', 'the left one', 'the middle one', 'the bottom one', 'the top one'],
        'options_yes_no': ['yes', 'no', 'not sure'],
    },

    'L1_LVQA': {
        'which_object': ['square', 'circle', 'triangle', 'not sure', 'no available option', 'not sure'],
        'what_shape': ['Triangular pyramid', 'Cone', 'Cube', 'Sphere', 'None', 'not sure'],
        # 'how_many': [0, 1, 2, 3, 4, 5, 6, 7, 8],
        'what_movement_2d': ['horizontal', 'inclined', 'vertical', 'no movenment', 'None', 'not sure'],
        'what_movement_3d': ['Rotation', 'Shrinking', 'Translation', 'Enlarging', 'None', 'not sure'],
        'what_surface': ['Rough', 'Moderate', 'Smooth', 'None', 'not sure'],
        'spacial_change': ['Bottom-left to top-right', 'Bottom-right to top-left', 'Top-left to bottom-right', 'Top-right to bottom-left', 'None', 'not sure', 'No movement',],
        'options_yes_no': ['yes', 'no', 'not sure'],
    },

    'L2_MAIC': {
        'maic_cls_list': maic_cls_list,
        'prompt_maic': "There may be one or more sound-emitting objects in the provided audio. \nPlease strictly output the answer in the format answer={'cls_1': count_1, 'cls_2': count_2}, \nfor example, answer={'dog': 2, 'cat': 3, 'male': 1}. \n"
    },

    'L2_MVIC': {
        'mvic_cls_list': mvic_cls_list,
        'prompt_mvic': "There may be one or more visible objects in the provided image. \nPlease strictly output the answer in the format answer={'cls_1': count_1, 'cls_2': count_2}, \nfor example, answer={'dog': 2, 'cat': 3, 'male': 1}. \n Possible categoris are in the list: mvic_cls_list"
    },

    'L3_AVH': {
        'prompt_avh': "Please answer the question based on the given audio and video.",
        'avh_options_list': ['yes', 'no', 'not sure'],
    },

    'L3_VAH': {
        'prompt_vah': "Please answer the question based on the given audio and video.",
        'vah_options_list': ['yes', 'no', 'not sure'],
    },

    'L3_AVL': {
        'prompt_avl': prompt_avl,
        'avl_cls_list': avl_cls_list,
    },

    'L3_AVM': {
        'prompt_avm': 'Please answer the question based on the given audio and video.',
        'avm_options_list': ['yes', 'no', 'not sure'],
    },

    'L3_AVR': {
        'prompt_avr': "Please output the indices of the images list, starting from 0. For example: [], or [0, 3], or [1, 4, 9]."
    },

    'L3_VAR': {
        'prompt_var': "Please output the indices of the wavs list, starting from 0. For example: [], or [0, 3], or [1, 4, 9]."
    },

    'L4_AVC': {

    },

    'L4_AVLG': {
        'prompt_avlg': prompt_avlg,
    },

    'L4_AVQA': {
        'avqa_options_list_is': ['yes', 'no', 'not sure'],
    },

    'L5_AVLG': {
        'prompt_avlg': prompt_avlg,
    },

    'L5_AVQA': {
        'avqa_cls_list': avqa_cls_list,
        'avqa_options_list_is': ['yes', 'no', 'not sure'],
    },

    'L6_AVSQA': {
        'avsqa_options_list_object': ['cube', 'pyramid', 'cone', 'sphere', 'no availabel option', 'not sure'],
        'avsqa_options_list_color': ['red', 'blue', 'white', 'black', 'gray', 'green', 'no availabel option', 'not sure'],
        'prompt_avsqa': "You may choose multiple options; separated by semicolons. Your answer should be enclosed within ##, for example: #your_ans#.",
        'options_yes_no': ['yes', 'no', 'not sure'],
    }
}
def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args

def get_real_path(task_path: str, src_path: str) -> str:
    """传入taskpath和一些文件的path，构造文件的真实path

    Args:
        task_path (str): task path
        src_path (str): 每个文件的path

    Returns:
        str: 文件的真实path
    """
    temp_path = os.path.join(task_path, src_path)
    return os.path.normpath(temp_path)

def get_real_options_or_classes(d: dict) -> str:
    """Replace pseudo-options with real options text."""
    opts = d['input']['question'].get('options')
    if opts in havib_constants.get(d['task'], {}):
        opts = havib_constants[d['task']][opts]
    if opts:
        label = 'semantic categories' if 'cls' in opts else 'options'
        return f"Available {label} are: {opts}"
    return ''

def get_real_prompt(d: dict) -> str:
    """Replace pseudo-prompt with real prompt text."""
    prm = d['input']['question'].get('prompt')
    if prm in havib_constants.get(d['task'], {}):
        prm = havib_constants[d['task']][prm]
    return prm or ''

def get_real_input(d: dict) -> str:
    """Concatenate prompt, options, and question text into one input string."""
    prompt = get_real_prompt(d)
    options = get_real_options_or_classes(d)
    question = d['input']['question']['text'] or ''
    # 去掉多余的句点
    parts = [p for p in (prompt, options, question) if p]
    return " ".join(parts)
def concat_audio(audio_paths: List[str]) -> str:
    """
    Concatenate multiple audio files into one WAV file.
    Returns the path to the temp WAV file.
    """
    clips = [AudioFileClip(p) for p in audio_paths]
    final = concatenate_audioclips(clips)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    out_path = tmp.name
    final.write_audiofile(out_path, fps=16000, logger=None)
    return out_path

def images_to_video(image_paths: List[str], duration: float, fps: int = 1) -> str:
    """
    Turn a list of images into a silent video of total `duration` seconds.
    Each image is shown for `duration / len(image_paths)` seconds.
    Returns the path to the temp MP4 file.
    """
    single_dur = duration / len(image_paths)
    clips = [ImageClip(p).set_duration(single_dur) for p in image_paths]
    video = concatenate_videoclips(clips, method="compose")
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    out_path = tmp.name
    video.write_videofile(out_path, fps=fps, codec="libx264", audio=False, logger=None)
    return out_path

def images_and_audio_to_video(image_paths: List[str], audio_paths: List[str], fps: int = 1) -> str:
    """
    Concatenate audio_paths into one audio, then build a video from image_paths
    that matches the audio duration, and merge them.
    Returns the path to the temp MP4 file.
    """
    # 1) build the concatenated audio
    audio_path = concat_audio(audio_paths)
    audio_clip = AudioFileClip(audio_path)
    # 2) build video from images matching audio duration
    duration = audio_clip.duration
    vid_path = images_to_video(image_paths, duration, fps=fps)
    # 3) attach audio to video
    video_clip = AudioFileClip(audio_path)  # re-open to avoid MoviePy caching issues
    from moviepy.editor import VideoFileClip
    base_vid = VideoFileClip(vid_path)
    final = base_vid.set_audio(audio_clip)
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    out_path = tmp.name
    final.write_videofile(out_path, fps=fps, codec="libx264", logger=None)
    return out_path 

def extract_audio_from_video(video_path: str) -> str:
    """Extract audio track from a video file and write it to a temp WAV."""
    clip = VideoFileClip(video_path)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    out_path = tmp.name
    clip.audio.write_audiofile(out_path, fps=16000, logger=None)
    clip.reader.close(); clip.audio.reader.close_proc()
    return out_path

def generate_silent_wav(video_path: str, fps: int = 16000) -> str:
    """
    Generate a silent WAV file matching the duration of the input video.
    Returns the path to the temporary WAV file.
    """
    # 1. 读取视频以获取时长
    clip = VideoFileClip(video_path)
    duration = clip.duration

    # 2. 创建一个每一时刻都返回 0.0（静音）的 AudioClip
    silent = AudioClip(lambda t: 0.0, duration=duration, fps=fps)

    # 3. 写出到临时文件
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    out_wav = tmp.name
    silent.write_audiofile(out_wav, fps=fps, logger=None)

    # 4. 关闭视频文件避免资源泄露
    clip.reader.close()
    return out_wav

def build_conversation(text: str) -> list:
    """
    Return a standard conversation list given the user's question text.
    """
    return [
        {"from": "human", "value": text},
        {"from": "gpt",   "value": "None"}
    ]
    

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load arguments
args = parse_args()
args = Config(args).config

# Load the list of test files
all_decode_info = args.all_decode_info

# Set the decoder output directory
decode_root = os.path.dirname(args.delta_ckpt_path)
current_time = datetime.now()
timestamp = current_time.strftime("%Y%m%d%H%M")
decode_root = os.path.join(decode_root, timestamp)
os.makedirs(decode_root, exist_ok=True)
OmegaConf.save(args, os.path.join(decode_root, "config.yaml"))

# Initialise the model
ds_engine = OpenLLAMAPEFTModel(**args)
delta_ckpt = torch.load(args['delta_ckpt_path'], map_location=torch.device('cpu'))
ds_engine.load_state_dict(delta_ckpt, strict=False)
ds_engine = ds_engine.eval().half().to(device)

# Load test data as a list of dataloaders
dataloader_lst = []
for modality, task, task_path in all_decode_info:
    print("Loading data from: {}".format(task_path))
    task_name = f"L{task_path.rsplit('/', 1)[0][-1]}_{task_path.rsplit('/', 1)[-1]}"
    model_name = "video_salmonn"
    save_prediction_json = f'/share/nlp/tuwenming/projects/HAVIB/eval/user_outputs/{model_name}/tasks/{task_name}.json'
    os.makedirs(os.path.dirname(save_prediction_json), exist_ok=True)
    print('>>> save res to:', save_prediction_json)

    data_json_path = os.path.join(task_path, "data.json")
    with open(data_json_path, "r", encoding='utf-8') as f:
        raw_data = json.load(f)
    print(">>>Finished load raw data...")
    parsed_data = []
    for item in raw_data:
        inp = item.get('input', {})
        question = inp.get('question', {})
        entry = {
            'id': item.get('id'),
            'task': item.get('task'),
            'subtask': item.get('subtask', None),
            'text': get_real_input(item),
            'audio_list': inp.get('audio_list', None),
            'image_list': inp.get('image_list', None),
            'video': inp.get('video', None)
        }
        parsed_data.append(entry)

    print(">>>Finished parse raw data...")    
    dummy_audio = "./dummy/1272-128104-0000.flac"
    new_data = []
    # for data in tqdm(parsed_data):
    #     _id = data['id']
    #     _task = data['task']
    #     _subtask = data['subtask']
    #     text = data['text']
    #     audio_list = (
    #         [get_real_path(task_path, p) for p in data["audio_list"]]
    #         if data["audio_list"] else None
    #     )
    #     image_list = (
    #         [get_real_path(task_path, p) for p in data["image_list"]]
    #         if data["image_list"] else None
    #     )
    #     video = (
    #         get_real_path(task_path, data['video'])
    #         if data['video'] else None
    #     )

    #     if audio_list and not image_list and not video:
    #         # Case 1: 仅音频, "image_name": "音频地址",
    #         media = concat_audio(audio_list) if len(audio_list)>1 else audio_list[0]
            
                
    #     elif image_list and not audio_list and not video:
    #         # Case 2: 仅图像 
    #         """
    #             "image_name": [
    #         "./dummy/761183272.jpg", # image 地址
    #         "./dummy/1272-128104-0000.flac" # dummy audio
    #     ]
    #         """
    #         media = [image_list[0], dummy_audio]
            

    #     elif video and not audio_list and not image_list:
    #         # Case 3: 仅视频
    #         """
    #         "image_name": [
    #         "./dummy/4405327307.mp4",
    #         "./dummy/4405327307.wav" # 生成一个无声音频与视频对齐
    #     ]
    #         """ 
    #         silent_wav = generate_silent_wav(video)
    #         media = [video, silent_wav]
            

    #     elif video and audio_list:
    #         # Case 4: 视频+音频列表（实际上直接使用视频自带音频）
    #         """
    #         "image_name": [
    #         "./dummy/4405327307.mp4",
    #         "./dummy/4405327307.wav" 视频的音频
    #     ]
    #         """ 
    #         media = [video, audio_list[0]]
            

    #     elif image_list and audio_list and not video:
    #         # Case 5: 图像+音频 -> 合成视频
    #         vid = images_and_audio_to_video(image_list, audio_list, fps=1)
    #         media = [vid, audio_list[0]]
    #         """
    #         "image_name": [
    #         "./dummy/4405327307.mp4",
    #         "./dummy/4405327307.wav" # 需要根据视频获取wav
    #     ]
    #         """ 

    #     else:
    #         raise ValueError(f"Unsupported input combination for id={_id}")
            
    
        # conv = build_conversation(text)
        # new_item = {
        #     "id":           _id,
        #     "task":         _task,
        #     "subtask":      _subtask,
        #     "image_name":   media,
        #     "conversation": conv
        # }
        # print(">>> data=:", new_item)
        # new_data.append(new_item)
    
    
    
    data_path = os.path.join(task_path, "salmonn_data.json")
    with open(data_path, 'r', encoding='utf-8') as fout:
        new_data = json.load(fout)
        
        
    if modality == "audio":
        dataset = SupervisedAudioVisualDataset4Test(
            'audio',
            audio_data_path=data_path,
            use_whisper=args["use_whisper"],
            training=False,
            sin_pos=args["sin_pos"],
            return_raw=args["return_raw"],
            cache_dir=args["cache_dir"],
        )
    elif modality == "audioimage":
        dataset = SupervisedAudioVisualDataset4Test(
            'audioimage',
            audio_data_path="./dummy/dummy_audio.json",
            image_data_path=data_path,
            use_whisper=args["use_whisper"],
            training=False,
            sin_pos=args["sin_pos"],
            return_raw=args["return_raw"],
            cache_dir=args["cache_dir"],
        )
    elif modality == "audiovideoimage":
        dataset = SupervisedAudioVisualDataset4Test(
            'audiovideoimage',
            audio_data_path="./dummy/dummy_audio.json",
            video_data_path=data_path,
            use_whisper=args["use_whisper"],
            training=False,
            sin_pos=args["sin_pos"],
            return_raw=args["return_raw"],
            cache_dir=args["cache_dir"],
        )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args['batch_size'],
        num_workers=3,
        shuffle=False,
        collate_fn=dataset.collate,
        drop_last=False
    )
   
    # 3) 推理并收集 pred_records
    preds = []
    for batch in tqdm(dataloader, desc=f"Decoding {task_name}"):
        with torch.no_grad():
            # `generate` 返回一个列表，对应当前 batch 中每个样本的预测
            outputs = ds_engine(batch, generate=True)
        print('>>> ans=:', outputs)
        preds.extend(outputs)

    # 4) 结合 new_data 中的 id/task/subtask 构造 pred_records
    pred_records = [
        {
            "id":       entry["id"],
            "task":     entry["task"],
            "subtask":  entry.get("subtask"),
            "predict":  pred
        }
        for entry, pred in zip(new_data, preds)
    ]

    # 5) 写入当前数据集的预测结果 JSON
    with open(save_prediction_json, "w", encoding="utf-8") as fout:
        json.dump(pred_records, fout, ensure_ascii=False, indent=2)

    print(f">>> Wrote {len(pred_records)} records to {save_prediction_json}")