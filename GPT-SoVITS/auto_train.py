####输入接口
inp_filelist_path="/data/docker/liujing04/dataset/sxc1min/test_urls_zh.list"#入口是一个训练集标注文件
with open(inp_filelist_path,"r")as f:lines=f.read().strip("\n").split("\n")
#以下为工程同学可修改的参数
tmp_dataset_download_root="/data/docker/liujing04/tmp"#保存语音训练集的临时目录
language="ZH"#JA日语，ZH中文
gpu_id="0"#用来CUDA_VISIBLE_DEVICES的
is_half=True#半精度
all_parts=2#数据集预处理每一步的进程数,如果爆显存就调1
cmd_python="/root/miniconda3/envs/py39webui/bin/python"
log_dir="/data/docker/liujing04/tmp/logs"
batch_size_sovits=6#爆显存就调小
batch_size_gpt=4#爆显存就调小
########以下为内部逻辑
###1-数据集下载到本地
import os,json,yaml
import requests as r
s=r.session()
s.trust_env=False
from subprocess import Popen
import logging,logging.handlers
logging.getLogger("numba").setLevel(logging.WARNING)
log = logging.getLogger('werkzeug')
formatter = logging.Formatter('%(asctime)s\tline:%(lineno)d\t\t%(message)s',"%Y-%m-%d %H:%M:%S")
logger1 = logging.getLogger(__name__+"auto_main")
logger1.setLevel(level=logging.INFO)
log_dir="%s/auto_main"%log_dir
os.makedirs(log_dir,exist_ok=True)
handler1 = logging.handlers.TimedRotatingFileHandler("%s/auto_main.log"%log_dir, when='D', interval=1, backupCount=365)
handler1.setLevel(logging.INFO)
handler1.setFormatter(formatter)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logger1.addHandler(handler1)
logger1.addHandler(console)
logger1_info=logger1.info
log.addHandler(handler1)
log.addHandler(console)

model_name=inp_filelist_path.split("/")[-1].split(".")[0]
tmp_dataset_download_dir="%s/%s"%(tmp_dataset_download_root,model_name)
os.makedirs(log_dir,exist_ok=True)
os.makedirs(tmp_dataset_download_dir,exist_ok=True)
tmp_dataset_filelist="%s/%s.list"%(tmp_dataset_download_root,model_name)
opt=[]
for line in lines:
    url,name,text=line.split("|")
    resp=s.get(url)
    if(resp.status_code!=200):
        logger1_info("%s-%s"%(url,resp.content))
    else:
        with open("%s/%s"%(tmp_dataset_download_dir,name),"wb")as f:f.write(resp.content)
    opt.append("%s|%s|%s|%s"%(name,model_name,language,text))
#TTS标准训练集格式
with open(tmp_dataset_filelist,"w")as f:f.write("\n".join(opt))

###2-数据集预处理有3个子步骤
opt_dir='logs/%s'%model_name
base_config={
    'inp_text': tmp_dataset_filelist,
    'inp_wav_dir': tmp_dataset_download_dir,
    'exp_name': model_name,
    'opt_dir': opt_dir,
    'all_parts': str(all_parts),
    '_CUDA_VISIBLE_DEVICES': gpu_id,
    'is_half': str(is_half),
    'log_dir': str(log_dir)
}
#step1
ps=[]
for i in range(all_parts):
    exec_config=base_config.copy()
    exec_config["i_part"]=str(i)
    exec_config["bert_pretrained_dir"]="GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
    os.environ.update(exec_config)
    cmd='"%s" GPT_SoVITS/prepare_datasets/1-get-text.py'%cmd_python
    logger1_info(exec_config)
    logger1_info(cmd)
    p = Popen(cmd, shell=True)
    ps.append(p)
for p in ps:
    p.wait()
#step2
ps=[]
for i in range(all_parts):
    exec_config=base_config.copy()
    exec_config["i_part"]=str(i)
    exec_config["cnhubert_base_dir"]="GPT_SoVITS/pretrained_models/chinese-hubert-base"
    os.environ.update(exec_config)
    cmd='"%s" GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py'%cmd_python
    logger1_info(exec_config)
    logger1_info(cmd)
    p = Popen(cmd, shell=True)
    ps.append(p)
for p in ps:
    p.wait()
#step3
ps=[]
for i in range(all_parts):
    exec_config=base_config.copy()
    exec_config["i_part"]=str(i)
    exec_config["pretrained_s2G"]="GPT_SoVITS/pretrained_models/s2G488k.pth"
    exec_config["s2config_path"]="GPT_SoVITS/configs/s2.json"
    os.environ.update(exec_config)
    cmd='"%s" GPT_SoVITS/prepare_datasets/3-get-semantic.py'%cmd_python
    logger1_info(exec_config)
    logger1_info(cmd)
    p = Popen(cmd, shell=True)
    ps.append(p)
for p in ps:
    p.wait()
#合并step1和step3的子进程结果小文件到大文件
opt = []
for i_part in range(all_parts):
    txt_path = "%s/2-name2text-%s.txt" % (opt_dir, i_part)
    with open(txt_path, "r", encoding="utf8") as f:
        opt += f.read().strip("\n").split("\n")
    os.remove(txt_path)
path_text = "%s/2-name2text.txt" % opt_dir
with open(path_text, "w", encoding="utf8") as f:
    f.write("\n".join(opt) + "\n")
opt = ["item_name\tsemantic_audio"]
path_semantic = "%s/6-name2semantic.tsv" % opt_dir
for i_part in range(all_parts):
    semantic_path = "%s/6-name2semantic-%s.tsv" % (opt_dir, i_part)
    with open(semantic_path, "r", encoding="utf8") as f:
        opt += f.read().strip("\n").split("\n")
    os.remove(semantic_path)
with open(path_semantic, "w", encoding="utf8") as f:
    f.write("\n".join(opt) + "\n")
###3-train-step1(sovits)
SoVITS_weight_root="SoVITS_weights"
os.makedirs(SoVITS_weight_root,exist_ok=True)
with open("GPT_SoVITS/configs/s2.json") as f:
    data = f.read()
    data = json.loads(data)
s2_dir = 'logs/%s'%model_name
os.makedirs("%s/logs_s2" % (s2_dir), exist_ok=True)
data["train"]["batch_size"] = batch_size_sovits
data["train"]["epochs"] = 10
data["train"]["text_low_lr_rate"] = 0.4
data["train"]["pretrained_s2G"] = "GPT_SoVITS/pretrained_models/s2G488k.pth"
data["train"]["pretrained_s2D"] = "GPT_SoVITS/pretrained_models/s2D488k.pth"
data["train"]["if_save_latest"] = True
data["train"]["if_save_every_weights"] = True
data["train"]["save_every_epoch"] = 5
data["train"]["gpu_numbers"] = gpu_id
data["data"]["exp_dir"] = data["s2_ckpt_dir"] = s2_dir
data["save_weight_dir"] = SoVITS_weight_root
data["name"] = model_name
tmp_config_path = "%s/tmp_s2.json" % s2_dir
with open(tmp_config_path, "w") as f: f.write(json.dumps(data))

cmd = '"%s" GPT_SoVITS/s2_train.py --config "%s"' % (cmd_python, tmp_config_path)
logger1_info(cmd)
p_train_SoVITS = Popen(cmd, shell=True)
p_train_SoVITS.wait()
###4-train-step2(gpt)
GPT_weight_root="GPT_weights"
os.makedirs(GPT_weight_root,exist_ok=True)
with open("GPT_SoVITS/configs/s1longer.yaml") as f:
    data = f.read()
    data = yaml.load(data, Loader=yaml.FullLoader)
s1_dir = 'logs/%s'%model_name
os.makedirs("%s/logs_s1" % (s1_dir), exist_ok=True)
data["train"]["batch_size"] = batch_size_gpt
data["train"]["epochs"] = 15
data["pretrained_s1"] = "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
data["train"]["save_every_n_epoch"] = 5
data["train"]["if_save_every_weights"] = True
data["train"]["if_save_latest"] = True
data["train"]["if_dpo"] = False
data["train"]["half_weights_save_dir"] = GPT_weight_root
data["train"]["exp_name"] = model_name
data["train_semantic_path"] = "%s/6-name2semantic.tsv" % s1_dir
data["train_phoneme_path"] = "%s/2-name2text.txt" % s1_dir
data["output_dir"] = "%s/logs_s1" % s1_dir
os.environ["_CUDA_VISIBLE_DEVICES"] = gpu_id
os.environ["hz"] = "25hz"
tmp_config_path = "%s/tmp_s1.yaml" % s1_dir
with open(tmp_config_path, "w") as f: f.write(yaml.dump(data, default_flow_style=False))
cmd = '"%s" GPT_SoVITS/s1_train.py --config_file "%s" ' % (cmd_python, tmp_config_path)
logger1_info(cmd)
p_train_GPT = Popen(cmd, shell=True)
p_train_GPT.wait()