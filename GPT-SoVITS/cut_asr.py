###接口输入
inp_path="/data/docker/liujing04/家仆.mp3"#切割的输入长音频
opt_root="/data/docker/liujing04/test-cut-opt3"#切割后的短音频根目录
asr_opt_path="/data/docker/liujing04/asr-opt.list3"#三列，本地路径，文件名，asr的结果文本
###切割逻辑
import os,sys,numpy as np
import traceback
from scipy.io import wavfile
from tools.my_utils import load_audio
from slicer2 import Slicer

def slice(inp_path,opt_root,threshold=-34,min_length=4000,min_interval=300,hop_size=10,max_sil_kept=500,_max=0.9,alpha=0.25):
    os.makedirs(opt_root,exist_ok=True)
    slicer = Slicer(
        sr=32000,  # 长音频采样率
        threshold=      int(threshold),  # 音量小于这个值视作静音的备选切割点
        min_length=     int(min_length),  # 每段最小多长，如果第一段太短一直和后面段连起来直到超过这个值
        min_interval=   int(min_interval),  # 最短切割间隔
        hop_size=       int(hop_size),  # 怎么算音量曲线，越小精度越大计算量越高（不是精度越大效果越好）
        max_sil_kept=   int(max_sil_kept),  # 切完后静音最多留多长
    )
    _max=float(_max)
    alpha=float(alpha)
    try:
        name = os.path.basename(inp_path)
        audio = load_audio(inp_path, 32000)
        # print(audio.shape)
        for chunk, start, end in slicer.slice(audio):  # start和end是帧数
            tmp_max = np.abs(chunk).max()
            if(tmp_max>1):chunk/=tmp_max
            chunk = (chunk / tmp_max * (_max * alpha)) + (1 - alpha) * chunk
            wavfile.write(
                "%s/%s_%010d_%010d.wav" % (opt_root, name, start, end),
                32000,
                # chunk.astype(np.float32),
                (chunk * 32767).astype(np.int16),
            )
    except:
        print(traceback.format_exc())

slice(inp_path,opt_root)

###ASR逻辑
'''
pip install modelscope==1.10.0 funasr==0.8.7
ASR会先在~/.cache里面下载modelscope（国内CDN）的模型文件大概1G左右
'''
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
inference_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model=     'damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
    vad_model= 'damo/speech_fsmn_vad_zh-cn-16k-common-pytorch',
    punc_model='damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch',
)
opt_lines=[]
for name in os.listdir(opt_root):
    path="%s/%s"%(opt_root,name)
    text = inference_pipeline(audio_in=path)["text"]
    opt_lines.append("%s|%s|%s"%(path,name,text))
with open(asr_opt_path,"w")as f:f.write("\n".join(opt_lines))
