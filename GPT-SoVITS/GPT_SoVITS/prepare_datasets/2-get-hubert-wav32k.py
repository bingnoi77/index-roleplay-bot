# -*- coding: utf-8 -*-

import sys,os
inp_text=                           os.environ.get("inp_text")
inp_wav_dir=                        os.environ.get("inp_wav_dir")
exp_name=                           os.environ.get("exp_name")
i_part=                             os.environ.get("i_part")
all_parts=                          os.environ.get("all_parts")
os.environ["CUDA_VISIBLE_DEVICES"]= os.environ.get("_CUDA_VISIBLE_DEVICES")
from feature_extractor import cnhubert
opt_dir=                            os.environ.get("opt_dir")
cnhubert.cnhubert_base_path=                os.environ.get("cnhubert_base_dir")
is_half=eval(os.environ.get("is_half","True"))
log_dir = os.environ.get("log_dir","/data/log")

import pdb,traceback,numpy as np,logging
from scipy.io import wavfile
import librosa,torch
now_dir = os.getcwd()
sys.path.append(now_dir)
from my_utils import load_audio

import logging,logging.handlers
logging.getLogger("numba").setLevel(logging.WARNING)
log = logging.getLogger('werkzeug')
formatter = logging.Formatter('%(asctime)s\tline:%(lineno)d\t\t%(message)s',"%Y-%m-%d %H:%M:%S")
logger1 = logging.getLogger(__name__+"step2")
logger1.setLevel(level=logging.INFO)
log_dir="%s/dataset2"%log_dir
os.makedirs(log_dir,exist_ok=True)
handler1 = logging.handlers.TimedRotatingFileHandler("%s/step2.log"%log_dir, when='D', interval=1, backupCount=365)
handler1.setLevel(logging.INFO)
handler1.setFormatter(formatter)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logger1.addHandler(handler1)
logger1.addHandler(console)
logger1_info=logger1.info
# app.logger.addHandler(handler1)
log.addHandler(handler1)
log.addHandler(console)

from time import time as ttime
import shutil
def my_save(fea,path):#####fix issue: torch.save doesn't support chinese path
    dir=os.path.dirname(path)
    name=os.path.basename(path)
    # tmp_path="%s/%s%s.pth"%(dir,ttime(),i_part)
    tmp_path="%s%s.pth"%(ttime(),i_part)
    torch.save(fea,tmp_path)
    shutil.move(tmp_path,"%s/%s"%(dir,name))

hubert_dir="%s/4-cnhubert"%(opt_dir)
wav32dir="%s/5-wav32k"%(opt_dir)
os.makedirs(opt_dir,exist_ok=True)
os.makedirs(hubert_dir,exist_ok=True)
os.makedirs(wav32dir,exist_ok=True)

maxx=0.95
alpha=0.5
if torch.cuda.is_available():
    device = "cuda:0"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
model=cnhubert.get_model()
# is_half=False
if(is_half==True):
    model=model.half().to(device)
else:
    model = model.to(device)

nan_fails=[]
def name2go(wav_name,wav_path):
    hubert_path="%s/%s.pt"%(hubert_dir,wav_name)
    if(os.path.exists(hubert_path)):return
    tmp_audio = load_audio(wav_path, 32000)
    tmp_max = np.abs(tmp_audio).max()
    if tmp_max > 2.2:
        logger1_info("%s-filtered,%s" % (wav_name, tmp_max))
        return
    tmp_audio32 = (tmp_audio / tmp_max * (maxx * alpha*32768)) + ((1 - alpha)*32768) * tmp_audio
    tmp_audio32b = (tmp_audio / tmp_max * (maxx * alpha*1145.14)) + ((1 - alpha)*1145.14) * tmp_audio
    tmp_audio = librosa.resample(
        tmp_audio32b, orig_sr=32000, target_sr=16000
    )#不是重采样问题
    tensor_wav16 = torch.from_numpy(tmp_audio)
    if (is_half == True):
        tensor_wav16=tensor_wav16.half().to(device)
    else:
        tensor_wav16 = tensor_wav16.to(device)
    ssl=model.model(tensor_wav16.unsqueeze(0))["last_hidden_state"].transpose(1,2).cpu()#torch.Size([1, 768, 215])
    if np.isnan(ssl.detach().numpy()).sum()!= 0:
        nan_fails.append(wav_name)
        logger1_info("nan filtered:%s"%wav_name)
        return
    wavfile.write(
        "%s/%s"%(wav32dir,wav_name),
        32000,
        tmp_audio32.astype("int16"),
    )
    my_save(ssl,hubert_path )

with open(inp_text,"r",encoding="utf8")as f:
    lines=f.read().strip("\n").split("\n")

for line in lines[int(i_part)::int(all_parts)]:
    try:
        # wav_name,text=line.split("\t")
        wav_name, spk_name, language, text = line.split("|")
        if (inp_wav_dir != "" and inp_wav_dir != None):
            wav_name = os.path.basename(wav_name)
            wav_path = "%s/%s"%(inp_wav_dir, wav_name)

        else:
            wav_path=wav_name
            wav_name = os.path.basename(wav_name)
        name2go(wav_name,wav_path)
    except:
        logger1_info("%s-%s"%(line,traceback.format_exc()))

if(len(nan_fails)>0 and is_half==True):
    is_half=False
    model=model.float()
    for wav_name in nan_fails:
        try:
            name2go(wav_name)
        except:
            logger1_info("%s-%s"%(wav_name,traceback.format_exc()))
