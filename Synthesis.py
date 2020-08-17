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
    con=np.stack([con1,con2,D])
    return con

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
cons=[]
while i <len(words)-1:
    if words[i][1]!=words[i+1][0]:
        
        length2=words[i+1][0]-words[i][1]
        
        begin=last-10
        end=words[i][1]+length2
        
        length1=length2
        # print(begin,end)
        print('Part %d: from %d to %d len:%d n_note: %d'%(cot,begin,end,end-begin,i+1-last_n))
        con=gen(words[last_n:i+1],(begin,end,'-'))

        # print(con1)
        # print(con2)
        # print(D)

#         D=process_D(D)
        # print(D)
        cons.append(con)
        
        last=words[i+1][0]
        last_n=i+1
        cot+=1
    i+=1
    

length2=40
        
begin=last-10
end=words[i][1]+length2
        
length1=length2
print('Part %d: from %d to %d len:%d n_note: %d'%(cot,begin,end,end-begin,i+1-last_n))
con=gen(words[last_n:i+1],(begin,end,'-'))


# print(D)
cons.append(con)

os.system('rm ./tmp/cons/*')

for i in range(len(cons)):
#     print(cons[i].shape)
    np.save('./tmp/cons/%03d.npy'%i,cons[i])

os.environ['MKL_SERVICE_FORCE_INTEL']='true'
    
os.system('CUDA_VISIBLE_DEVICES=1 python synthesize.py')

