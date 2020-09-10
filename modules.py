import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
import numpy as np
import copy
import math

import hparams as hp
import utils

from transformer.Models import Encoder, Decoder

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device='cuda'

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class VarianceAdaptor(nn.Module):
    """ Variance Adaptor """

    def __init__(self):
        super(VarianceAdaptor, self).__init__()
        self.duration_predictor = LengthPredictor(n_layers=4)
        

        self.length_regulator = LengthRegulator()
        self.pitch_predictor = VariancePredictor(n_layers=4)
        self.energy_predictor = VariancePredictor(n_layers=4)
        self.relu=nn.ReLU()
        
#         self.pitch_bins = nn.Parameter(torch.exp(torch.linspace(np.log(hp.f0_min), np.log(hp.f0_max), hp.n_bins-1)))
#         self.energy_bins = nn.Parameter(torch.linspace(hp.energy_min, hp.energy_max, hp.n_bins-1))
        self.pitch_embedding = nn.Embedding(hp.n_bins*2, hp.encoder_hidden)
        self.energy_embedding = nn.Embedding(hp.n_bins*2, hp.encoder_hidden)
        
    
    def forward(self, x, src_seq, src_mask, mel_mask=None, duration_target=None, pitch_target=None, energy_target=None, max_len=None):
        log_duration_prediction = self.duration_predictor(x,src_seq, src_mask)
        if duration_target is not None:
            x, mel_len = self.length_regulator(x, duration_target, max_len)
        else:
            duration_rounded = torch.clamp(torch.round(log_duration_prediction), min=0)
            print(log_duration_prediction)
            x, mel_len = self.length_regulator(x, duration_rounded, max_len)
#             print(mel_len)
            mel_mask = utils.get_mask_from_lengths(mel_len)
#         print(mel_mask)
        pitch_prediction = self.relu(self.pitch_predictor(x, mel_mask))
#         print(pitch_prediction)
        if pitch_target is not None:
            src=torch.ceil((pitch_target-hp.f0_min)/(hp.f0_max-hp.f0_min)*hp.n_bins).long()
#             print(src)
            pitch_embedding = self.pitch_embedding(src)
        else:
            src=torch.ceil((pitch_prediction-hp.f0_min)/(hp.f0_max-hp.f0_min)*hp.n_bins).long()
            
            pitch_embedding = self.pitch_embedding(src)
        
        energy_prediction = self.energy_predictor(x, mel_mask)
        if energy_target is not None:
#             print(energy_target)
            src=torch.ceil((energy_target-hp.energy_min)/(hp.energy_max-hp.energy_min)*hp.n_bins).long()
#             print(src)
            energy_embedding = self.energy_embedding(src)
        else:
            src=torch.ceil((energy_prediction-hp.energy_min)/(hp.energy_max-hp.energy_min)*hp.n_bins).long()
            
            energy_embedding = self.energy_embedding(src)
        
        x = x + pitch_embedding + energy_embedding
        
        return x, log_duration_prediction, pitch_prediction, energy_prediction, mel_len, mel_mask


class LengthRegulator(nn.Module):
    """ Length Regulator """

    def __init__(self):
        super(LengthRegulator, self).__init__()

    def LR(self, x, duration, max_len):
        output = list()
        mel_len = list()
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        if max_len is not None:
            output = utils.pad(output, max_len)
        else:
            output = utils.pad(output)

        return output, torch.LongTensor(mel_len).to(device)

    def expand(self, batch, predicted):
        out = list()

        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(int(expand_size), -1))
        out = torch.cat(out, 0)

        return out

    def forward(self, x, duration, max_len):
        
#         print(duration)
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len


class VariancePredictor(nn.Module):
    """ Pitch and Energy Predictor """

    def __init__(self,n_layers):
        super(VariancePredictor, self).__init__()

        self.decoder = Decoder(n_layers)
        self.linear_layer = nn.Linear(hp.decoder_hidden, 1)

    def forward(self, encoder_output, mask):
        
        out = self.decoder(encoder_output,mask)
        out = self.linear_layer(out)
        out = out.squeeze(-1)
        
        if mask is not None:
            out = out.masked_fill(mask, 0.)
        
        return out

class LengthPredictor(nn.Module):
    """ Duration Predictor """

    def __init__(self,n_layers):
        super(LengthPredictor, self).__init__()

        self.decoder = Decoder(n_layers)
        self.linear_layer = nn.Linear(hp.decoder_hidden, 1)
        self.tanh=nn.Tanh()
        
    def forward(self, encoder_output,src_seq, mask):
#         print(mask.shape,encoder_output.shape)
        out = self.decoder(encoder_output,mask)
        out = self.linear_layer(out)
        out = out.squeeze(-1)
#         print(out)
        if mask is not None:
            out = out.masked_fill(mask, 0.)
        lengths=src_seq[:,:,2]
#         print(out.shape,lengths.shape)
        out=(self.tanh(out)+1.0)
        out=out*lengths
#         print(out)
        return out

class Conv(nn.Module):
    """
    Convolution Module
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=True,
                 w_init='linear'):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(in_channels,
                              out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=bias)

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)

        return x
