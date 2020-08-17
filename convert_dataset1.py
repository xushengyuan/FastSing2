import numpy as np
import pyworld as pw
import os
from tqdm import tqdm
import soundfile as sf

mel_dir='/ssd/data/mels/'
f0_dir='/ssd/data/f0/'
wav_dir='/ssd/data/split/'
mel_paths=os.listdir(mel_dir)
f0_paths=os.listdir(f0_dir)
for i in tqdm(range(len(mel_paths))):
    mel=np.load(mel_dir+mel_paths[i])
    f0=np.load(f0_dir+f0_paths[i]).astype(np.float64)
    f0=f0/5.0+40.0
    f0=440.0*2**((f0-69)/12)

    arr1=[]
    for i in range(mel.shape[0]):
        tmp=np.zeros(513)
        tmp.fill(512)
        x=np.concatenate([np.arange(512),tmp],axis=0)
#             print(x)
        arr1.append(np.interp(x,
                     np.linspace(0,512,128),
                     mel[i][32:])[np.newaxis,:])
    sp=np.concatenate(arr1,axis=0)
#     plt.matshow(sp)
#     plt.savefig('out_sp_%02d.png'%index)
#     plt.cla()

    arr2=[]
    for i in range(mel.shape[0]):
        arr2.append(np.interp(np.arange(1025),
                     np.linspace(0,1025,32),
                     mel[i][:32])[np.newaxis,:])
    ap=np.concatenate(arr2,axis=0)
#     plt.matshow(ap)
#     plt.savefig('out_ap.png')
    sp=np.exp(sp)
    ap=(ap+18.0)/20.0
    #     print(ap.max(),ap.min(),ap.mean())

#     print(f0.shape,sp.shape,ap.shape)
    length=min(f0.shape[0],sp.shape[0],ap.shape[0])
    f0=f0[:length]
    sp=sp[:length]
    ap=ap[:length]
    y = pw.synthesize(f0, sp, ap, 32000, 8.0)
    sf.write(wav_dir+mel_paths[i][:-3]+'wav',y,32000)