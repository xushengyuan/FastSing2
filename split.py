import numpy as np
import soundfile as sf
import os
import pyworld as pw
from tqdm import tqdm

def herz2note(x):  
    return 69+12*np.log(x/440.0)/np.log(2)  



dir='/ssd/mu_yao/wav'
files=os.listdir(dir)
#print(files)

lens=[]
for file in tqdm(files):
    tmp,sr=sf.read(os.path.join(dir,file))
    lens.append(tmp.shape[0])

base_dir='/data/wav_muyao'
dirs=os.listdir(base_dir)
dirs.sort()
for dir in ['b6']:
    if len(dir)!=2:
        continue
    print(dir)
    wav1,sr=sf.read(os.path.join(base_dir,dir,"all_01.wav"),dtype='int16')
    wav2,sr=sf.read(os.path.join(base_dir,dir,"all_02.wav"),dtype='int16')
#     f01, t=pw.dio(wav1[:32000*10]. astype(np.float64), 32000)
#     f01=herz2note(f01[f01.nonzero()].mean())
#     f02, t=pw.dio(wav2[:32000*10].astype(np.float64)
# , 32000)
#     f02=herz2note(f02[f02.nonzero()].mean())
#     print(dir, f01, f02)
    wav=np.append(wav1,wav2)
    len_=0
    for i in tqdm(range(len(lens))):
        sf.write(os.path.join('/ssd/mu_yao/wav/',dir+'-'+files[i]),wav[len_:len_+lens[i]],32000)
#         print(len_,len_+lens[i])
        len_+=lens[i]
    
