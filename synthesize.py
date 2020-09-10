import torch
import torch.nn as nn
import numpy as np
import os
import argparse
from tqdm import tqdm
import re
from string import punctuation
import soundfile as sf
# from g2p_en import G2p

from fastspeech2 import FastSpeech2
from text import text_to_sequence, sequence_to_text
import hparams as hp
import utils
import audio as Audio

    
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device='cuda'
def get_FastSpeech2(num):
    checkpoint_path = os.path.join(hp.checkpoint_path, "checkpoint_{}.pth.tar".format(num))
#     model = FastSpeech2()
    model = nn.DataParallel(FastSpeech2())

    model_data=torch.load(checkpoint_path)['model']
#     keys=model_data.keys()
#     model_data_={}
#     for key in keys:
#         model_data_[key[7:]]=model_data[key]
#     print(model_data_.keys())
    model.load_state_dict(model_data)
    model.requires_grad = False
    model.eval()
    return model

def synthesize(model, condition,index ):
     # long filename will result in OS Error
    
    src_len = torch.from_numpy(np.array([condition.shape[1]])).to(device)
    condition=condition[np.newaxis,:,:]
    condition=torch.LongTensor(condition).to(device).transpose(1,2)
#     print(condition)
    
    ap_output, sp_output, sp_postnet_output, log_duration_output, f0_output,energy_output, src_mask, ap_mask,sp_mask ,variance_adaptor_output,decoder_output= model(condition ,src_len)

    length = min(ap_output.shape[1],sp_output.shape[1],f0_output.shape[1])
    ap = ap_output[0, :length].detach().cpu().double().numpy()
    sp = sp_output[0, :length].detach().cpu().double().numpy()
    sp_postnet = sp_postnet_output[0, :length].detach().cpu().double().numpy()
    f0_output = f0_output[0, :length].detach().cpu().double().numpy()
    energy_output = energy_output[0, :length].detach().cpu().numpy()
    print(condition.transpose(1,2)[0][2])
    print(log_duration_output)
#     print(ap.shape,sp_postnet.shape,f0_output.shape)
#     return utils.world_infer()
#     y=untils.world_infer()
#     if not os.path.exists(hp.test_path):
#         os.makedirs(hp.test_path)

#     Audio.tools.inv_mel_spec(mel_postnet, os.path.join(hp.test_path, '{}_griffin_lim_{}.wav'.format(prefix, sentence)))
#     if hp.vocoder=='waveglow':
#         melgan = utils.get_melgan()
#         melgan.to(device)
#         utils.waveglow_infer(mel_postnet_torch, waveglow, os.path.join(hp.test_path, '{}_{}_{}.wav'.format(prefix, hp.vocoder, sentence)))
#     if hp.vocoder=='melgan':
#         waveglow = utils.get_waveglow()
#         waveglow.to(device)
#         utils.melgan_infer(mel_postnet_torch, melgan, os.path.join(hp.test_path, '{}_{}_{}.wav'.format(prefix, hp.vocoder, sentence)))
    y=utils.world_infer(ap,sp,f0_output)
    sp_postnet=np.swapaxes(sp_postnet,0,1)
    utils.plot_data([(sp_postnet, f0_output, energy_output)], ['Synthesized Spectrogram'], filename=os.path.join(hp.test_path, 'out_%03d.png'%index))
    return y

if __name__ == "__main__":
    # Test
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=int, default=376000)
    args = parser.parse_args()
    

    model = get_FastSpeech2(args.step).to(device)

    n=len(os.listdir('./tmp/cons'))
    wav=np.zeros([1])
    for index in tqdm(range(n)):
        condition=np.load("./tmp/cons/%03d.npy"%index)
        y=synthesize(model,condition,index)
        wav=np.append(wav,y)
        
    sf.write(os.path.join(hp.test_path, 'out_%08d.wav'%args.step),wav,32000)
