import numpy as np
import soundfile as sf
import os
from tqdm import tqdm

dir='/ssd/mu_yao/wav'
files=os.listdir(dir)
print(files)

data,sr=sf.read(os.path.join(dir,files[0]),dtype='int16')
for file in tqdm(files[1:]):
    tmp,sr=sf.read(os.path.join(dir,file),dtype='int16')
    data=np.append(data,tmp)
    
sf.write('/data/all.wav',data,32000)