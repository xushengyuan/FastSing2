import numpy as np
import os
import math
# from scipy.io.wavfile import read
import soundfile as sf
import pyworld as pw
import torch
import audio as Audio
from utils import get_alignment
from text import _clean_text
import hparams as hp
import random
from tqdm import tqdm



def prase_textgrid(lines):
    # print(lines[0].strip())
    # print(lines[1].strip())
    # assert lines[0].strip()=='File type = "ooTextFile"'
    # assert lines[1].strip()=='Object class = "TextGrid"'

    lines=lines[4:]
    file_length=float(lines[0].strip()[7:])

    lines=lines[2:]
    assert lines[0].strip()=='size = 5'

    lines=lines[2:]

    tiers=[]
    for i in range(5):
        # print(i)
        tier=[]
        lines=lines[5:]
        # print(lines[0].strip())
        n_interval=int(lines[0].strip()[18:])
        lines=lines[1:]

        for j in range(n_interval):
            lines=lines[1:]
            begin=float(lines[0].strip()[7:])
            end=float(lines[1].strip()[7:])
            text=lines[2].strip()[8:-1].strip()
            tier.append([begin,end,text])
            lines=lines[3:]

        tiers.append(tier)
    # print(tiers)
    return tiers,file_length
                
def build_from_path(in_dir, out_dir):
    index = 1
    train = list()
    val = list()
    f0_max = energy_max = 0
    f0_min = energy_min = 1000000
    n_frames = 0
    
    basenames=os.listdir(os.path.join(hp.data_path,'textgrid'))
    for basename in tqdm(basenames):

        basename=basename[:-9]
        try:
            info, f_max, f_min, e_max, e_min, n = process_utterance(in_dir, out_dir, basename)
#             print(info, f_max, f_min, e_max, e_min, n)
            if random.random()<hp.val_rate:
                val.append(info)
            else:
                train.append(info)
        except:
            print("ERROR!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",basename)
        
        index = index + 1

        f0_max = max(f0_max, f_max)
        f0_min = min(f0_min, f_min)
        energy_max = max(energy_max, e_max)
        energy_min = min(energy_min, e_min)
        n_frames += n
    
    with open(os.path.join(out_dir, 'stat.txt'), 'w', encoding='utf-8') as f:
        strs = ['Total time: {} hours'.format(n_frames*hp.hop_length/hp.sampling_rate/3600),
                'Total frames: {}'.format(n_frames),
                'Min F0: {}'.format(f0_min),
                'Max F0: {}'.format(f0_max),
                'Min energy: {}'.format(energy_min),
                'Max energy: {}'.format(energy_max)]
        for s in strs:
            print(s)
            f.write(s+'\n')
    
    return [r for r in train if r is not None], [r for r in val if r is not None]


def find_intervals(tier,begin,end):
    result=[]
    
    # print(tier,begin,end)
    for i in range(len(tier)):
        # print(begin,tier[i][0],tier[i][1],end)
        if (begin<=tier[i][0] and tier[i][1]<=end)\
         or(tier[i][0]<begin and begin<tier[i][1] and tier[i][1]<end)\
         or(begin<tier[i][0] and tier[i][0]<end and end<tier[i][1])\
         or(tier[i][0]<=begin and end<=tier[i][1]):
            result.append(i)
    return result


dict_v={}
dict_vc={}
dict_uvc={}
def prepare_dict():
    dict_v_in=open('dict_v.txt','r')
    dict_vc_in=open('dict_vc.txt','r')
    dict_uvc_in=open('dict_uvc.txt','r')
    lines_v=dict_v_in.readlines()
    lines_vc=dict_vc_in.readlines()
    lines_uvc=dict_uvc_in.readlines()

    for i in range(len(lines_v)):
        dict_v[lines_v[i].strip()]=i+1
    for i in range(len(lines_vc)):
        dict_vc[lines_vc[i].strip()]=len(lines_v)+i+1
    for i in range(len(lines_uvc)):
        dict_uvc[lines_uvc[i].strip()]=len(lines_v)+len(lines_vc)+i+1
        
prepare_dict()

def gen_musical_score(tiers):
    lens=[]
    for i in range(len(tiers[2])):
        interval=tiers[2][i]
        length=int((interval[1]-interval[0])*hp.sampling_rate/hp.hop_length)
        
        intervals=find_intervals(tiers[0],interval[0],interval[1])
        if len(intervals)==1:
            lens.append(length)
            continue
        for index in intervals:
            p=tiers[0][index]
            if p[2] in dict_uvc:
                length-=int((p[1]-p[0])*hp.sampling_rate/hp.hop_length)
                
        intervals=find_intervals(tiers[0],tiers[2][i+1][0],tiers[2][i+1][1])
        if len(intervals)>1:
            for index in intervals:
                p=tiers[0][index]
                if p[2] in dict_uvc:
                    length+=int((p[1]-p[0])*hp.sampling_rate/hp.hop_length)
        lens.append(length)
    
    lengths=[]
    phon=[]
    pitch=[]
#     print(dict_v,dict_vc,dict_uvc)
#     print(len(tiers[2]),len(lens))
    for interval in tiers[0]:
#         print(interval[2])
        if interval[2]=='sil' or interval[2]=='':
            phon.append(0)
            pitch.append(0)
            lengths.append(int((interval[1]-interval[0])*hp.sampling_rate/hp.hop_length))
            continue
        elif interval[2] in dict_v:
            phon.append(dict_v[interval[2]])
        elif interval[2] in dict_vc:
            phon.append(dict_vc[interval[2]])
        elif interval[2] in dict_uvc:
            phon.append(dict_uvc[interval[2]])
        else:
            print(interval)
        intervals=find_intervals(tiers[1],interval[0],interval[1])
        pitch.append(int(tiers[1][intervals[0]][2]))
        
        intervals=find_intervals(tiers[2],interval[0],interval[1])
#         print(intervals)
        lengths.append(lens[intervals[0]])
        
    
    
    length=np.array(lengths)
    phon=np.array(phon)
    pitch=np.array(pitch)
#     print(phon.shape,pitch.shape,length.shape)
    assert len(phon)==len(pitch)==len(length)
    condition=np.stack([phon,pitch,length])
#     print(condition)
    return condition
        

def process_utterance(in_dir, out_dir, basename):
    wav_path = os.path.join(in_dir, 'wav', '{}.wav'.format(basename))
    tg_path = os.path.join(in_dir, 'textgrid', '{}.TextGrid'.format(basename)) 
    
    # Get alignments
    textgrid_in=open(tg_path,'r', encoding='utf-16')
    lines=textgrid_in.readlines()
    tiers = prase_textgrid(lines)[0]

    duration=[]
#     print(tiers)
    for interval in tiers[0]:
#         print(interval)
        length=interval[1]-interval[0]
        duration.append(int(round(length*hp.sampling_rate/hp.hop_length)))
    duration=np.array(duration)
#     print(tiers[0])
    
    condition=gen_musical_score(tiers)
    
    
    # Read and trim wav files
    wav,fs = sf.read(wav_path)
    assert fs==hp.sampling_rate
    wav = wav.astype(np.float32)
    
    # Compute fundamental frequency
    _f0, t = pw.dio(wav.astype(np.float64), hp.sampling_rate, frame_period=hp.hop_length/hp.sampling_rate*1000)
    f0_herz = pw.stonemask(wav.astype(np.float64), _f0, t, hp.sampling_rate)
    
    f0_note=[]
    for i in range(len(f0_herz)):
        if f0_herz[i]==0:
            f0_note.append(0.0)
        else:
            f0_note.append(Audio.tools.herz2note(f0_herz[i]))
    f0=np.array(f0_note)
    
    # Compute mel-scale spectrogram and energy
    mel_spectrogram, energy = Audio.tools.get_mel_from_wav(torch.FloatTensor(wav))
    mel_spectrogram=mel_spectrogram.transpose(0,1).numpy()
    energy = np.log(energy.numpy().astype(np.float32))
    
    ap,sp=Audio.tools.get_target(wav.astype(np.float64),hp.sampling_rate,hp.n_ap_channels,hp.n_sp_channels)
    print(ap.shape,sp.shape)
    
    min_len=min([duration.sum(),f0.shape[0],energy.shape[0],mel_spectrogram.shape[0],ap.shape[0],sp.shape[0]])
    duration[-1]-=duration.sum()-min_len
    f0=f0[:min_len]
    energy=energy[:min_len]
    mel_spectrogram=mel_spectrogram[:min_len]
    ap=ap[:min_len]
    sp=sp[:min_len]
#     print(condition.shape,duration.shape,duration.sum(),f0.shape,energy.shape,mel_spectrogram.shape,ap.shape,sp.shape)
    
#     assert 
    assert duration.sum()==f0.shape[0]==energy.shape[0]==mel_spectrogram.shape[0]==ap.shape[0]==sp.shape[0]

    con_filename = '{}-condition-{}.npy'.format(hp.dataset, basename)
    np.save(os.path.join(out_dir, 'condition', con_filename), condition, allow_pickle=False)
    
    # Save alignment
    ali_filename = '{}-ali-{}.npy'.format(hp.dataset, basename)
    np.save(os.path.join(out_dir, 'alignment', ali_filename), duration, allow_pickle=False)

    # Save fundamental prequency
    f0_filename = '{}-f0-{}.npy'.format(hp.dataset, basename)
    np.save(os.path.join(out_dir, 'f0', f0_filename), f0, allow_pickle=False)

    # Save energy
    energy_filename = '{}-energy-{}.npy'.format(hp.dataset, basename)
    np.save(os.path.join(out_dir, 'energy', energy_filename), energy, allow_pickle=False)

    # Save spectrogram

    ap_filename = '{}-ap-{}.npy'.format(hp.dataset, basename)
    np.save(os.path.join(out_dir, 'ap', ap_filename), ap, allow_pickle=False)
    
    sp_filename = '{}-sp-{}.npy'.format(hp.dataset, basename)
    np.save(os.path.join(out_dir, 'sp', sp_filename), sp, allow_pickle=False)
    mel_filename = '{}-mel-{}.npy'.format(hp.dataset, basename)
    np.save(os.path.join(out_dir, 'mel', mel_filename), mel_spectrogram, allow_pickle=False)
 
    return basename, max(f0), min([f for f in f0 if f != 0]), max(energy), min(energy), mel_spectrogram.shape[0]
