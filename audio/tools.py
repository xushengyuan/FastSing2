import torch
import numpy as np
from scipy.io.wavfile import read
from scipy.io.wavfile import write

import audio.stft as stft
from audio.audio_processing import griffin_lim
import hparams
import pyworld as pw

_stft = stft.TacotronSTFT(
    hparams.filter_length, hparams.hop_length, hparams.win_length,
    hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
    hparams.mel_fmax)


def herz2note(x):
    return 69+12*np.log(x/440.0)/np.log(2)


def note2herz(n):
    return 440.0*2**((n-69)/12)

def get_target(x,fs,n_ap_channels,n_sp_channels,f0):
    _f0, t = pw.dio(x,fs, f0_floor=75.0, f0_ceil=1000.0,
                    frame_period=8.0)
    f0_herz = f0[:_f0.shape[0]]
    f0_herz[_f0<1.0]=0.0
    sp = pw.cheaptrick(x, f0_herz, t, fs)
    ap = pw.d4c(x, f0_herz, t, fs)
    # print(sp.shape)

    # plt.matshow(ap)
    # plt.show()
    ap=ap*20-18
    arr=[]
    for i in range(sp.shape[0]):
        arr.append(np.interp(np.linspace(0,1025,n_ap_channels),np.arange(1025),ap[i])[np.newaxis,:])
    _ap=np.concatenate(arr,axis=0)

    sp=np.log(sp)
    # plt.matshow(sp)
    # plt.show()
    arr=[]
    for i in range(sp.shape[0]):
        arr.append(np.interp(np.linspace(0,1025,n_sp_channels),np.arange(1025),sp[i])[np.newaxis,:])
    _sp=np.concatenate(arr,axis=0)

    
#     mel=mel+20.0
#     mel=np.where(mel>0,mel,0)
#     mel=mel/mel.max()
#     plt.matshow(mel)
#     plt.show()

    return _ap,_sp,f0_herz

def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def get_mel(filename):
    audio, sampling_rate = load_wav_to_torch(filename)
    if sampling_rate != _stft.sampling_rate:
        raise ValueError("{} {} SR doesn't match target {} SR".format(
            sampling_rate, _stft.sampling_rate))
    audio_norm = audio / hparams.max_wav_value
    audio_norm = audio_norm.unsqueeze(0)
    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
    melspec, energy = _stft.mel_spectrogram(audio_norm)
    melspec = torch.squeeze(melspec, 0)
    energy = torch.squeeze(energy, 0)
    # melspec = torch.from_numpy(_normalize(melspec.numpy()))

    return melspec, energy


def get_mel_from_wav(audio):
    sampling_rate = hparams.sampling_rate
    if sampling_rate != _stft.sampling_rate:
        raise ValueError("{} {} SR doesn't match target {} SR".format(
            sampling_rate, _stft.sampling_rate))
    audio_norm = audio / hparams.max_wav_value
    audio_norm = audio_norm.unsqueeze(0)
    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
    melspec, energy = _stft.mel_spectrogram(audio_norm)
    melspec = torch.squeeze(melspec, 0)
    energy = torch.squeeze(energy, 0)

    return melspec, energy


def inv_mel_spec(mel, out_filename, griffin_iters=60):
    mel = torch.stack([mel])
    # mel = torch.stack([torch.from_numpy(_denormalize(mel.numpy()))])
    mel_decompress = _stft.spectral_de_normalize(mel)
    mel_decompress = mel_decompress.transpose(1, 2).data.cpu()
    spec_from_mel_scaling = 1000
    spec_from_mel = torch.mm(mel_decompress[0], _stft.mel_basis)
    spec_from_mel = spec_from_mel.transpose(0, 1).unsqueeze(0)
    spec_from_mel = spec_from_mel * spec_from_mel_scaling

    audio = griffin_lim(torch.autograd.Variable(
        spec_from_mel[:, :, :-1]), _stft.stft_fn, griffin_iters)

    audio = audio.squeeze()
    audio = audio.cpu().numpy()
    audio_path = out_filename
    write(audio_path, hparams.sampling_rate, audio)
