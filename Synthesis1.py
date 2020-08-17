import numpy as np
import os
import time
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
import soundfile as sf
from parsevsqx import vsqx2notes
import random
import torch
from stft import TacotronSTFT

dict_size=437
fs = 32000
n=49
wav_path="D:\wav"
tg_path="./data/TextGrid"
dict_path="./pinyin.txt"
mel_ground_truth = "./data/mels"
condition1='./data/con1s'
condition2='./data/con2s'
alignment_path = "./data/alignments"

pinyin={}
def prepare_dict():
    fin=open(dict_path,'r')
    lines=fin.readlines()
    for i in range(len(lines)):
        pinyin[lines[i].strip()]=i+1
pinyin_=['']
def prepare_dict4():
    fin=open(dict_path,'r')
    lines=fin.readlines()
    for i in range(len(lines)):
        pinyin_.append(lines[i].strip())
prepare_dict4()
def pad_words(words,sentence):
    # print(words)
    # print(sentence)
    if sentence[0]<words[0][0] :
        words=[[sentence[0],words[0][0],'',64]]+words
    if sentence[1]>words[-1][1]:
        words=words+[[words[-1][1],sentence[1],'',64]]
    # print(words)
    return words
        
def get_D(words):
    D=[]
    # print(words)
    for i in range(len(words)):
        length=words[i][1]-words[i][0]
        D.append(int(length))
    return np.array(D)

phon_dict={}
def prepare_dict2():
    dict_in=open('dict.txt','r')
    lines=dict_in.readlines()

    for line in lines:
        phons=line.strip().split(' ')
        phon_dict[phons[0]]=phons[1:]

    # print(phon_dict)
prepare_dict2()

dict_v={}
dict_vc={}
dict_uvc={}
def prepare_dict3():
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
        
prepare_dict3()



def get_con1(words):
    # print(words)
    con1=[]
    for i in range(len(words)):
        if words[i][2] in pinyin:
            con1.append(pinyin[words[i][2]])
        else:
            con1.append(0)
    return np.array(con1)

def get_con2(words):
    con2=[]
    for i in range(len(words)):
        con2.append(words[i][3])
    return np.array(con2)

def gen(notes,sentence):
    notes=pad_words(notes,sentence)
    D=get_D(notes)
    con1=get_con1(notes)    
    con2=get_con2(notes)

    # print(D)
    # print(mel.shape,D.sum(),D.shape,con1.shape,con2.shape)
    # os.system('pause')


    # print(D)
    # print(mel.shape,D.sum(),D.shape,con1.shape,con2.shape)
    # print(con1)
    # print(con2)
    # os.system('pause')
    assert D.shape[0]==con1.shape[0]==con2.shape[0]
    # assert mel.shape[0]==D.sum()
    return [con1,con2,D]

def process_D(D):
    for i in range(len(D)):
        D[i]=int(D[i]+6*random.random()-3)
    return D

stft = TacotronSTFT()

def get_mel(audio):
    audio_norm = torch.FloatTensor(audio.astype(np.float64))
    audio_norm = audio_norm.unsqueeze(0)
    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
    melspec = stft.mel_spectrogram(audio_norm)
    melspec = torch.squeeze(melspec, 0).transpose(0,1)
    return melspec.numpy()

def find_phon(phon):
    if phon=='sil' or phon=='':
        return 0
    elif phon in dict_v:
        return dict_v[phon]
    elif phon in dict_vc:
        return dict_vc[phon]
    elif phon in dict_uvc:
        return dict_uvc[phon]
    

def convert(src):
#     print(src.shape)
    length=src.shape[1]
    words=[]
    for i in range(length):
        if src[0][i]!=1:
            words.append([[src[x][i]for x in range(3)]])
        else:
            words[-1].append([src[x][i]for x in range(3)])
    intervals=[]
#     print(pinyin_)
    for word in words:
#         length_word=word[0][2]
#         for j in range(1,len(word)):
#             length_word+=word[j][2]
#         print(word)
        if word[0][0]==0:
            intervals.append([0,0,word[0][2]])
            continue
        _pinyin=pinyin_[word[0][0]]
        phons=phon_dict[_pinyin]
        phon_n=[find_phon(phon)for phon in phons]
        intervals.append([phon_n[0],word[0][1],word[0][2]])
        if len(phon_n)==2:
            intervals.append([phon_n[1],word[0][1],word[0][2]])
        for j in range(1,len(word)):
            if len(phon_n)==2:
                intervals.append([phon_n[1],word[j][1],word[j][2]])
            else:
                intervals.append([phon_n[0],word[j][1],word[j][2]])

    return np.array(intervals)
        
prepare_dict()
# main()

words,begin,end = vsqx2notes(sys.argv[1])

# x, _fs = sf.read(sys.argv[2])
# refer_mel=get_mel(x)

wav=np.zeros(1)

length1=20
last=begin
last_n=0
cot=1
i=0
con1s=[]
con2s=[]
Ds=[]
while i <len(words)-1:
    if words[i][1]!=words[i+1][0]:
        
        length2=words[i+1][0]-words[i][1]
        
        begin=last-10
        end=words[i][1]+length2
        
        length1=length2
        # print(begin,end)
        print('Part %d: from %d to %d len:%d n_note: %d'%(cot,begin,end,end-begin,i+1-last_n))
        con1,con2,D=gen(words[last_n:i+1],(begin,end,'-'))

        # print(con1)
        # print(con2)
        # print(D)

#         D=process_D(D)
        # print(D)
        con1s.append(con1)
        con2s.append(con2)
        Ds.append(D)
        
        last=words[i+1][0]
        last_n=i+1
        cot+=1
    i+=1
    

length2=40
        
begin=last-10
end=words[i][1]+length2
        
length1=length2
print('Part %d: from %d to %d len:%d n_note: %d'%(cot,begin,end,end-begin,i+1-last_n))
con1,con2,D=gen(words[last_n:i+1],(begin,end,'-'))

# D=process_D(D)
# print(D)
con1s.append(con1)
con2s.append(con2)
Ds.append(D)

os.system('rm ./tmp/cons/*')
# os.system('rm ./tmp/con2s/*')
# os.system('rm ./tmp/Ds/*')
# print(con1s[0].shape,con2s[0].shape,Ds[0].shape)
for i in range(len(con1s)):
    
    con=np.stack([con1s[i],con2s[i],Ds[i]])
    print(con)
    _con=np.swapaxes(convert(con),0,1)
    print(_con)
    np.save('./tmp/cons/%03d.npy'%i,_con)
#     np.save('./tmp/con2s/%03d.npy'%i,con2s[i])
#     np.save('./tmp/Ds/%03d.npy'%i,Ds[i])
#     np.save('./tmp/refer_mels/%03d.npy'%i,refer_mel)

os.environ['MKL_SERVICE_FORCE_INTEL']='true'
    
os.system('CUDA_VISIBLE_DEVICES=0 python synthesize.py')

