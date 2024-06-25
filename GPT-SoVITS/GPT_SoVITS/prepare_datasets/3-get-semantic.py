import os

inp_text = os.environ.get("inp_text")
exp_name = os.environ.get("exp_name")
i_part = os.environ.get("i_part")
all_parts = os.environ.get("all_parts")
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("_CUDA_VISIBLE_DEVICES")
opt_dir = os.environ.get("opt_dir")
pretrained_s2G = os.environ.get("pretrained_s2G")
s2config_path = os.environ.get("s2config_path")
is_half = eval(os.environ.get("is_half", "True"))
log_dir = os.environ.get("log_dir","/data/log")
import math, traceback
import multiprocessing
import sys, pdb

now_dir = os.getcwd()
sys.path.append(now_dir)
from random import shuffle
import torch.multiprocessing as mp
from glob import glob
from tqdm import tqdm
import logging, librosa, utils, torch
from module.models import SynthesizerTrn

import logging,logging.handlers
logging.getLogger("numba").setLevel(logging.WARNING)
log = logging.getLogger('werkzeug')
formatter = logging.Formatter('%(asctime)s\tline:%(lineno)d\t\t%(message)s',"%Y-%m-%d %H:%M:%S")
logger1 = logging.getLogger(__name__+"step1")
logger1.setLevel(level=logging.INFO)
log_dir="%s/dataset1"%log_dir
os.makedirs(log_dir,exist_ok=True)
handler1 = logging.handlers.TimedRotatingFileHandler("%s/step1.log"%log_dir, when='D', interval=1, backupCount=365)
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


hubert_dir = "%s/4-cnhubert" % (opt_dir)
semantic_path = "%s/6-name2semantic-%s.tsv" % (opt_dir, i_part)
if os.path.exists(semantic_path) == False:
    os.makedirs(opt_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    hps = utils.get_hparams_from_file(s2config_path)
    vq_model = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model
    )
    if is_half == True:
        vq_model = vq_model.half().to(device)
    else:
        vq_model = vq_model.to(device)
    vq_model.eval()
    # utils.load_checkpoint(utils.latest_checkpoint_path(hps.s2_ckpt_dir, "G_*.pth"), vq_model, None, True)
    # utils.load_checkpoint(pretrained_s2G, vq_model, None, True)
    logger1_info(
        vq_model.load_state_dict(
            torch.load(pretrained_s2G, map_location="cpu")["weight"], strict=False
        )
    )

    def name2go(wav_name, lines):
        hubert_path = "%s/%s.pt" % (hubert_dir, wav_name)
        if os.path.exists(hubert_path) == False:
            return
        ssl_content = torch.load(hubert_path, map_location="cpu")
        if is_half == True:
            ssl_content = ssl_content.half().to(device)
        else:
            ssl_content = ssl_content.to(device)
        codes = vq_model.extract_latent(ssl_content)
        semantic = " ".join([str(i) for i in codes[0, 0, :].tolist()])
        lines.append("%s\t%s" % (wav_name, semantic))

    with open(inp_text, "r", encoding="utf8") as f:
        lines = f.read().strip("\n").split("\n")

    lines1 = []
    for line in lines[int(i_part) :: int(all_parts)]:
        # print(line)
        try:
            # wav_name,text=line.split("\t")
            wav_name, spk_name, language, text = line.split("|")
            wav_name = os.path.basename(wav_name)
            # name2go(name,lines1)
            name2go(wav_name, lines1)
        except:
            logger1_info("%s-%s"%(line, traceback.format_exc()))
    with open(semantic_path, "w", encoding="utf8") as f:
        f.write("\n".join(lines1))
