import torch
import torch.nn as nn
import numpy as np
import os
import argparse
import re
from string import punctuation
from g2p_en import G2p

from fastspeech2 import FastSpeech2
from text import text_to_sequence, sequence_to_text
import hparams as hp
import utils
import audio as Audio

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device='cuda'
def get_FastSpeech2(num):
    checkpoint_path = os.path.join(hp.checkpoint_path, "checkpoint_{}.pth.tar".format(num))
    model = nn.DataParallel(FastSpeech2())
    model.load_state_dict(torch.load(checkpoint_path)['model'])
    model.requires_grad = False
    model.eval()
    return model

def synthesize(model, waveglow, melgan, text, sentence, prefix=''):
    sentence = sentence[:200] # long filename will result in OS Error
    
    src_len = torch.from_numpy(np.array([text.shape[1]])).to(device)
        
    mel, mel_postnet, log_duration_output, f0_output, energy_output, _, _, mel_len = model(text, src_len)
    
    mel_torch = mel.transpose(1, 2).detach()
    mel_postnet_torch = mel_postnet.transpose(1, 2).detach()
    mel = mel[0].cpu().transpose(0, 1).detach()
    mel_postnet = mel_postnet[0].cpu().transpose(0, 1).detach()
    f0_output = f0_output[0].detach().cpu().numpy()
    energy_output = energy_output[0].detach().cpu().numpy()

    if not os.path.exists(hp.test_path):
        os.makedirs(hp.test_path)

    Audio.tools.inv_mel_spec(mel_postnet, os.path.join(hp.test_path, '{}_griffin_lim_{}.wav'.format(prefix, sentence)))
    if waveglow is not None:
        utils.waveglow_infer(mel_postnet_torch, waveglow, os.path.join(hp.test_path, '{}_{}_{}.wav'.format(prefix, hp.vocoder, sentence)))
    if melgan is not None:
        utils.melgan_infer(mel_postnet_torch, melgan, os.path.join(hp.test_path, '{}_{}_{}.wav'.format(prefix, hp.vocoder, sentence)))
    
    utils.plot_data([(mel_postnet.numpy(), f0_output, energy_output)], ['Synthesized Spectrogram'], filename=os.path.join(hp.test_path, '{}_{}.png'.format(prefix, sentence)))


if __name__ == "__main__":
    # Test
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=int, default=30000)
    args = parser.parse_args()
    

    model = get_FastSpeech2(args.step).to(device)
    melgan = waveglow = None
    if hp.vocoder == 'melgan':
        melgan = utils.get_melgan()
        melgan.to(device)
    elif hp.vocoder == 'waveglow':
        waveglow = utils.get_waveglow()
        waveglow.to(device)
    
    for sentence in sentences:
        text = preprocess(sentence)
        synthesize(model, waveglow, melgan, text, sentence, prefix='step_{}'.format(args.step))
