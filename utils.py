import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pyworld as pw
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from scipy.io import wavfile
import os
import soundfile as sf

import text
import hparams as hp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device='cpu'
def get_alignment(tier):
    sil_phones = ['sil', 'sp', 'spn']

    phones = []
    durations = []
    start_time = 0
    end_time = 0
    end_idx = 0
    for t in tier._objects:
        s, e, p = t.start_time, t.end_time, t.text

        # Trimming leading silences
        if phones == []:
            if p in sil_phones:
                continue
            else:
                start_time = s
        if p not in sil_phones:
            phones.append(p)
            end_time = e
            end_idx = len(phones)
        else:
            phones.append(p)
        durations.append(int(e*hp.sampling_rate/hp.hop_length)-int(s*hp.sampling_rate/hp.hop_length))

    # Trimming tailing silences
    phones = phones[:end_idx]
    durations = durations[:end_idx]
    
    return phones, durations, start_time, end_time

def process_meta(meta_path):
    with open(meta_path, "r", encoding="utf-8") as f:
        name = []
        for line in f.readlines():
            n = line.strip('\n')
            if os.path.isfile(os.path.join(hp.preprocessed_path,"{}-{}.npz".format(hp.dataset, n))):
                name.append(n)
        return name

def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param

def plot_data(data, titles=None, filename=None):
    fig, axes = plt.subplots(len(data), 1, squeeze=False)
    if titles is None:
        titles = [None for i in range(len(data))]

    def add_axis(fig, old_ax, offset=0):
        ax = fig.add_axes(old_ax.get_position(), anchor='W')
        ax.set_facecolor("None")
        return ax

    for i in range(len(data)):
        spectrogram, pitch, pitch_norm, energy = data[i]
#         spectrogram=np.swapaxes(spectrogram,0,1)
        axes[i][0].imshow(spectrogram, origin='lower')
        axes[i][0].set_aspect(2.5, adjustable='box')
        axes[i][0].set_ylim(0, hp.n_mel_channels)
        axes[i][0].set_title(titles[i], fontsize='medium')
        axes[i][0].tick_params(labelsize='x-small', left=False, labelleft=False) 
        axes[i][0].set_anchor('W')
        
        ax1 = add_axis(fig, axes[i][0])
        ax1.plot(pitch_norm, color='red', alpha=0.5)
        ax1.plot(pitch, color='tomato')
        ax1.set_xlim(0, spectrogram.shape[1])
        ax1.set_ylim(hp.f0_min, hp.f0_max)
        ax1.set_ylabel('F0', color='tomato')
        ax1.tick_params(labelsize='x-small', colors='tomato', bottom=False, labelbottom=False)
        
        
        
        ax2 = add_axis(fig, axes[i][0], 1.2)
        ax2.plot(energy, color='darkviolet')
        ax2.set_xlim(0, spectrogram.shape[1])
        ax2.set_ylim(hp.energy_min, hp.energy_max)
        ax2.set_ylabel('Energy', color='darkviolet')
        ax2.yaxis.set_label_position('right')
        ax2.tick_params(labelsize='x-small', colors='darkviolet', bottom=False, labelbottom=False, left=False, labelleft=False, right=True, labelright=True)
    
    plt.savefig(filename, dpi=200)
    plt.close()


def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(device)
    mask = (ids >= lengths.unsqueeze(1).expand(-1, max_len))

    return mask

def get_waveglow():
    waveglow = None
    return waveglow

def waveglow_infer(mel, waveglow, path):
    with torch.no_grad():
        wav = waveglow.infer(mel, sigma=1.0) * hp.max_wav_value
        wav = wav.squeeze().cpu().numpy()
    wav = wav.astype('int16')
    wavfile.write(path, hp.sampling_rate, wav)

def melgan_infer(mel, melgan, path):
    with torch.no_grad():
        wav = melgan.inference(mel).cpu().numpy()
    wav = wav.astype('int16')
    wavfile.write(path, hp.sampling_rate, wav)

    
def world_infer(ap,sp,f0):
#     print(ap.shape,sp.shape,f0.shape)

    f0=np.where(f0<=41,0.0,f0)
    f0=440.0*2**((f0-69)/12)

    arr1=[]
    for i in range(sp.shape[0]):
        x=np.arange(1025)
        
        y_hat=np.interp(x,
                     np.linspace(0,1025,128),
                     sp[i])
        arr1.append(y_hat)
    sp=np.stack(arr1)
    
#     print(sp)
    
    arr1_=[]
    for i in range(sp.shape[1]):
        x=np.arange(sp.shape[0]*2)
        y_hat=np.interp(x,
                        np.linspace(0,sp.shape[0]*2,sp.shape[0]),
                       sp[:,i])
        arr1_.append(y_hat)
    sp=np.swapaxes(np.stack(arr1_),0,1)
#     print(sp)
    sp=np.exp(sp)
#     print(sp)
    
    arr2=[]
    for i in range(ap.shape[0]):
        arr2.append(np.interp(np.arange(1025),
                     np.linspace(0,1025,32),
                     ap[i]))
    ap=np.stack(arr2)
    
#     print(ap)
#     plt.matshow(ap)
#     plt.savefig('out_ap.png')
    
    
    arr2_=[]
    for i in range(ap.shape[1]):
        x=np.arange(ap.shape[0]*2)
        y_hat=np.interp(x,
                        np.linspace(0,ap.shape[0]*2,ap.shape[0]),
                       ap[:,i])
        arr2_.append(y_hat)
    ap=np.swapaxes(np.stack(arr2_),0,1)
    
    
    
    f0=np.interp(np.arange(f0.shape[0]*2),np.linspace(0,f0.shape[0]*2,f0.shape[0]),f0)
#         ap=(ap+18.0)/20.0
    #     print(ap.max(),ap.min(),ap.mean())

#     print(f0.shape,sp.shape,ap.shape)
    print(f0.shape,sp.shape,ap.shape)
    
    length=min(f0.shape[0],sp.shape[0],ap.shape[0])
    f0=f0[:length]
    sp=sp[:length]
    ap=ap[:length]
#     print(f0.shape,sp.shape,ap.shape)
    f0=f0.astype(np.float64).copy(order='C')
#     print(f0)
    sp=sp.astype(np.float64).copy(order='C')
#     print(sp)
    ap=ap.astype(np.float64).copy(order='C')
#     print(ap)
    y = pw.synthesize(f0,sp ,ap , 32000, 4.0)
#     sf.write(path,y,32000)
    return y
# sp=np.load('/ssd/mu_yao/preprocessed/fastsing_dataset/sp/fastsing_dataset-sp-012_13.npy')
# ap=np.load('/ssd/mu_yao/preprocessed/fastsing_dataset/ap/fastsing_dataset-ap-012_13.npy')
# f0=np.load('/ssd/mu_yao/preprocessed/fastsing_dataset/f0/fastsing_dataset-f0-012_13.npy')
# y=world_infer(ap,sp,f0,'out.wav')
def get_melgan():
    melgan = torch.hub.load('seungwonpark/melgan', 'melgan')
    melgan.eval()
    return melgan

def pad_1D(inputs, PAD=0):

    def pad_data(x, length, PAD):
        x_padded = np.pad(x, (0, length - x.shape[0]),
                          mode='constant',
                          constant_values=PAD)
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded

def pad_2D(inputs, maxlen=None):

    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(x, (0, max_len - np.shape(x)[0]),
                          mode='constant',
                          constant_values=PAD)
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output

def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0)for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len-batch.size(0)), "constant", 0.0)
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len-batch.size(0)), "constant", 0.0)
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded
