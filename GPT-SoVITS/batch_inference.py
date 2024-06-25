import sys,os#
import traceback

os.environ["CUDA_VISIBLE_DEVICES"]="3"
sys.path.append("/data/docker/liujing04/gpt-vits/github/GPT-SoVITS-main/GPT_SoVITS")
import argparse
import os
import pdb
import signal
import sys
from time import time as ttime
import torch
import librosa
import soundfile as sf
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
import uvicorn
from transformers import AutoModelForMaskedLM, AutoTokenizer
import numpy as np
from feature_extractor import cnhubert
from io import BytesIO
from module.models import SynthesizerTrn
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from text import cleaned_text_to_sequence
from text.cleaner import clean_text
from module.mel_processing import spectrogram_torch
from my_utils import load_audio
import config as global_config
from inference_webui import *
import requests
import numpy as np
import base64
from scipy.io import wavfile

g_config = global_config.Config()
g_config.sovits_path="/data/docker/liujing04/gpt-vits/github/sovits_weights/yunfei_e8_s272.pth"
g_config.gpt_path="/data/docker/liujing04/gpt-vits/github/gpt_weights/yunfei-e15.ckpt"

# AVAILABLE_COMPUTE = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description="GPT-SoVITS api")

parser.add_argument("-s", "--sovits_path", type=str, default=g_config.sovits_path, help="SoVITS模型路径")
parser.add_argument("-g", "--gpt_path", type=str, default=g_config.gpt_path, help="GPT模型路径")

parser.add_argument("-dr", "--default_refer_path", type=str, default="",
                    help="默认参考音频路径, 请求缺少参考音频时调用")
parser.add_argument("-dt", "--default_refer_text", type=str, default="", help="默认参考音频文本")
parser.add_argument("-dl", "--default_refer_language", type=str, default="", help="默认参考音频语种")

parser.add_argument("-d", "--device", type=str, default=g_config.infer_device, help="cuda / cpu")
parser.add_argument("-p", "--port", type=int, default=g_config.api_port, help="default: 9880")
parser.add_argument("-a", "--bind_addr", type=str, default="127.0.0.1", help="default: 127.0.0.1")
parser.add_argument("-fp", "--full_precision", action="store_true", default=False, help="覆盖config.is_half为False, 使用全精度")
parser.add_argument("-hp", "--half_precision", action="store_true", default=False, help="覆盖config.is_half为True, 使用半精度")
# bool值的用法为 `python ./api.py -fp ...`
# 此时 full_precision==True, half_precision==False

parser.add_argument("-hb", "--hubert_path", type=str, default=g_config.cnhubert_path, help="覆盖config.cnhubert_path")
parser.add_argument("-b", "--bert_path", type=str, default=g_config.bert_path, help="覆盖config.bert_path")

args = parser.parse_args()

sovits_path = args.sovits_path
gpt_path = args.gpt_path

default_refer_path = args.default_refer_path
default_refer_text = args.default_refer_text
default_refer_language = args.default_refer_language
has_preset = False

device = args.device
port = args.port
host = args.bind_addr

if sovits_path == "":
    sovits_path = g_config.pretrained_sovits_path
    print(f"[WARN] 未指定SoVITS模型路径, fallback后当前值: {sovits_path}")
if gpt_path == "":
    gpt_path = g_config.pretrained_gpt_path
    print(f"[WARN] 未指定GPT模型路径, fallback后当前值: {gpt_path}")

# 指定默认参考音频, 调用方 未提供/未给全 参考音频参数时使用
if default_refer_path == "" or default_refer_text == "" or default_refer_language == "":
    default_refer_path, default_refer_text, default_refer_language = "", "", ""
    print("[INFO] 未指定默认参考音频")
    has_preset = False
else:
    print(f"[INFO] 默认参考音频路径: {default_refer_path}")
    print(f"[INFO] 默认参考音频文本: {default_refer_text}")
    print(f"[INFO] 默认参考音频语种: {default_refer_language}")
    has_preset = True

is_half = g_config.is_half
if args.full_precision:
    is_half = False
if args.half_precision:
    is_half = True
if args.full_precision and args.half_precision:
    is_half = g_config.is_half  # 炒饭fallback

print(f"[INFO] 半精: {is_half}")

cnhubert_base_path = args.hubert_path
bert_path = args.bert_path

cnhubert.cnhubert_base_path = cnhubert_base_path
tokenizer = AutoTokenizer.from_pretrained(bert_path)
bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
if is_half:
    bert_model = bert_model.half().to(device)
else:
    bert_model = bert_model.to(device)


def get_bert_feature(text, word2ph):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)  #####输入是long不用管精度问题，精度随bert_model
        res = bert_model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
    assert len(word2ph) == len(text)
    phone_level_feature = []
    for i in range(len(word2ph)):
        repeat_feature = res[i].repeat(word2ph[i], 1)
        phone_level_feature.append(repeat_feature)
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    # if(is_half==True):phone_level_feature=phone_level_feature.half()
    return phone_level_feature.T


n_semantic = 1024
dict_s2 = torch.load(sovits_path, map_location="cpu")
hps = dict_s2["config"]
print(hps)

class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")


hps = DictToAttrRecursive(hps)
hps.model.semantic_frame_rate = "25hz"
dict_s1 = torch.load(gpt_path, map_location="cpu")
config = dict_s1["config"]
ssl_model = cnhubert.get_model()
if is_half:
    ssl_model = ssl_model.half().to(device)
else:
    ssl_model = ssl_model.to(device)

vq_model = SynthesizerTrn(
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    **hps.model)
if is_half:
    vq_model = vq_model.half().to(device)
else:
    vq_model = vq_model.to(device)
vq_model.eval()
print(vq_model.load_state_dict(dict_s2["weight"], strict=False))
hz = 50
max_sec = config['data']['max_sec']
t2s_model = Text2SemanticLightningModule(config, "ojbk", is_train=False)
t2s_model.load_state_dict(dict_s1["weight"])
if is_half:
    t2s_model = t2s_model.half()
t2s_model = t2s_model.to(device)
t2s_model.eval()
total = sum([param.nelement() for param in t2s_model.parameters()])
print("Number of parameter: %.2fM" % (total / 1e6))


def get_spepc(hps, filename):
    audio = load_audio(filename, int(hps.data.sampling_rate))
    audio = torch.FloatTensor(audio)
    audio_norm = audio
    audio_norm = audio_norm.unsqueeze(0)
    spec = spectrogram_torch(audio_norm, hps.data.filter_length, hps.data.sampling_rate, hps.data.hop_length,
                             hps.data.win_length, center=False)
    return spec


# dict_language = {
#     "中文": "zh",
#     "英文": "en",
#     "日文": "ja",
#     "ZH": "zh",
#     "EN": "en",
#     "JA": "ja",
#     "zh": "zh",
#     "en": "en",
#     "ja": "ja"
# }

def get_tts_wavs(ref_wav_path, prompt_text, prompt_language, textss, text_language, how_to_cut="凑四句一切", top_k=5, top_p=1, temperature=1, ref_free = False):
    audios_opt=[]
    for texts in textss:###硬来，不复用共通部分了，懒得写。。
        sr,audio=next(get_tts_wav(ref_wav_path, prompt_text, prompt_language, texts, text_language, how_to_cut="凑四句一切", top_k=5, top_p=1, temperature=1, ref_free = False))
        audios_opt.append([texts,audio])
    return audios_opt


# with open(r"D:\BaiduNetdiskDownload\gsv\美女test.txt","r",encoding="utf8")as f:
#     textss=f.read().strip("\n").split("\n")
# for idx,(text,audio)in enumerate(get_tts_wavs(r"D:\BaiduNetdiskDownload\gsv\美女.wav_109808000_109941440 (enhanced).wav", "我从来都不是，走偶像包袱路线的。", "中文", textss, "中文")):
#     print(idx,text)
#     sf.write(r"D:\BaiduNetdiskDownload\gsv\美女输出\美女测试\48k-无dpo-e10-更好\%04d-%s.wav"%(idx,text),audio,32000)

###单次离线测试
text="一个男人半夜遇上抢劫，却反过来将小混混打得落花流水，并且从他们身上得到了一件宝物。这个宝物能够让他看到人体内的秘密，他因此来到了一家古玩店，却发现这件宝物连二十都不值。就在他准备换一家古玩店的时候，紫色的提示框出现在他的眼前，告诉他问题出在佛像内部。"

sr,audio=next(get_tts_wav("/data/docker/liujing04/gpt-vits/github/wavs/云飞.mp3_12782080_12902720.wav", "我先是一愣，随即一脸微笑的主动迎了上去，想要成为冥界之主。", "中文", text, "中文", how_to_cut="凑四句一切", top_k=5, top_p=1, temperature=1, ref_free = False))
wavfile.write("save-test.wav",32000,audio.astype("int16"))#####咋没切呢？？？


from flask import Flask, request, jsonify
import base64
from my_utils import load_audio

app = Flask(__name__)
@app.route('/tts', methods=['POST'])
def tts():
    try:
        # 假设客户端以JSON格式发送音频数据，键为'audio_data'
        t0=ttime()
        text = request.json['text']
        sr, audio = next(get_tts_wav("/data/docker/liujing04/gpt-vits/github/wavs/云飞.mp3_12782080_12902720.wav", "我先是一愣，随即一脸微笑的主动迎了上去，想要成为冥界之主。", "中文", text, "中文", how_to_cut="凑四句一切", top_k=5, top_p=1, temperature=1, ref_free=False))
        # pdb.set_trace()
        print(audio.max())
        audio_bytes = audio.tobytes()
        audio_data_str = base64.b64encode(audio_bytes).decode('utf-8')
        t1=ttime()
        print(t1-t0)
        return jsonify({"message": "Success.","audio":audio_data_str})
    except:
        return jsonify({"message":traceback.format_exc(),"audio":""})

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=4567)
