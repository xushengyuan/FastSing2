import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer.Models import Encoder, Decoder
from transformer.Layers import PostNet
from modules import VarianceAdaptor
from utils import get_mask_from_lengths
import matplotlib.pyplot as plt
import hparams as hp
# from GST import GST

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device='cpu'
class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    def __init__(self, use_postnet=True):
        super(FastSpeech2, self).__init__()
        
#         self.gst = GST()
        self.encoder = Encoder()
        self.variance_adaptor = VarianceAdaptor()
        
        self.encoder_linear = nn.Linear(hp.encoder_hidden, hp.decoder_hidden)
        
        self.decoder = Decoder()
        
        if hp.vocoder=='WORLD':
#             self.f0_decoder= Decoder()
            self.ap_linear = nn.Linear(hp.decoder_hidden, hp.n_ap_channels)
            self.sp_linear = nn.Linear(hp.decoder_hidden, hp.n_sp_channels)
        else:
            self.mel_linear = nn.Linear(hp.decoder_hidden, hp.n_mel_channels)
        
        self.use_postnet = use_postnet
        if self.use_postnet:
            self.postnet = PostNet()

    def forward(self, src_seq, src_len, mel_len=None, d_target=None, p_target=None, p_norm=None, e_target=None, max_src_len=None, max_mel_len=None):
#         print(src_seq.shape)
#         print(src_len.shape)
        src_mask = get_mask_from_lengths(src_len, max_src_len)
#         print(src_mask.shape)
        mel_mask = get_mask_from_lengths(mel_len, max_mel_len) if mel_len is not None else None
        if hp.vocoder=='WORLD':
            ap_mask = get_mask_from_lengths(mel_len, max_mel_len) if mel_len is not None else None
            sp_mask = get_mask_from_lengths(mel_len, max_mel_len) if mel_len is not None else None


#         print(src_seq)
        encoder_output = self.encoder(src_seq, src_mask)
#         style_embed = self.gst(ref_mel)  # [N, 256]
#         style_embed = style_embed.expand_as(encoder_output)
#         encoder_output= encoder_output+style_embed
        encoder_output= encoder_output

        variance_adaptor_output, d_prediction, p_prediction, e_prediction, mel_len, mel_mask = self.variance_adaptor(
                    encoder_output, src_seq, src_mask, mel_mask, d_target, p_target, p_norm, e_target, max_mel_len)
#         print( variance_adaptor_output.shape)
#         plt.matshow( variance_adaptor_output[0].detach().cpu().numpy())
#         plt.savefig('variance_adaptor_output.png')
#         plt.cla()
#         print(mel_mask)
       # encoder_linear_output = self.encoder_linear(variance_adaptor_output)
        decoder_output = self.decoder(variance_adaptor_output, mel_mask)
#         print(sp_mask[0])
#         if hp.vocoder=='WORLD':
#             f0_decoder_output = self.f0_decoder(variance_adaptor_output, mel_mask)

        
        if hp.vocoder=='WORLD':
            ap_output = self.ap_linear(decoder_output)
            sp_output = self.sp_linear(decoder_output)


            if self.use_postnet:
                sp_output_postnet = self.postnet(sp_output) + sp_output
            else:
                sp_output_postnet = sp_output

            return ap_output, sp_output, sp_output_postnet, d_prediction, p_prediction, e_prediction, src_mask, ap_mask, sp_mask,encoder_output,variance_adaptor_output,decoder_output

        else:
            mel_output = self.mel_linear(decoder_output)

            if self.use_postnet:
                mel_output_postnet = self.postnet(mel_output) + mel_output
            else:
                mel_output_postnet = mel_output

            return mel_output, mel_output_postnet, d_prediction, p_prediction, e_prediction, src_mask, mel_mask, mel_len


if __name__ == "__main__":
    # Test
    model = FastSpeech2(use_postnet=True)
    print(model)
    print(sum(param.numel() for param in model.parameters()))
