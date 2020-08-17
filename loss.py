import torch
import torch.nn as nn
import hparams as hp

class FastSpeech2Loss(nn.Module):
    """ FastSpeech2 Loss """

    def __init__(self):
        super(FastSpeech2Loss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

    def forward(self, log_d_predicted, log_d_target, p_predicted, p_target, e_predicted, e_target, mel_output=None, mel_postnet_output=None, mel_target=None,ap_output=None, sp_output=None, sp_postnet_output=None, ap_target=None,sp_target=None,src_mask=None, mel_mask=None,ap_mask=None,sp_mask=None):
        log_d_target.requires_grad = False
        p_target.requires_grad = False
        e_target.requires_grad = False
        if hp.vocoder=='WORLD':
            ap_target.requires_grad = False
            sp_target.requires_grad = False
#             print(p_target)
#            print(src_mask)
#             print(sp_mask)
#             print(sp_output.shape,sp_target.shape)
            #print(log_d_predicted,log_d_target)
            #log_d_predicted = log_d_predicted.masked_select(src_mask)
            #log_d_target = log_d_target.masked_select(src_mask)
            #p_predicted = p_predicted.masked_select(sp_mask)
            #p_target = p_target.masked_select(sp_mask)
            #e_predicted = e_predicted.masked_select(sp_mask)
            #e_target = e_target.masked_select(sp_mask)
        else:
            mel_target.requires_grad = False
            log_d_predicted = log_d_predicted.masked_select(src_mask)
            log_d_target = log_d_target.masked_select(src_mask)
            p_predicted = p_predicted.masked_select(mel_mask)
            p_target = p_target.masked_select(mel_mask)
            e_predicted = e_predicted.masked_select(mel_mask)
            e_target = e_target.masked_select(mel_mask)
        
        

        d_loss = self.mae_loss(log_d_predicted, log_d_target)
        #print(log_d_predicted,log_d_target)
        p_loss = self.mae_loss(p_predicted, p_target)
#         print(p_predicted[0],p_target[0])
        e_loss = self.mae_loss(e_predicted, e_target)
        #print(e_predicted,e_target)
        
        if hp.vocoder=='WORLD':
#             sp = sp_output.masked_select(sp_mask.unsqueeze(-1))
#             ap = ap_output.masked_select(ap_mask.unsqueeze(-1))
#             sp_postnet_output = sp_postnet_output.masked_select(sp_mask.unsqueeze(-1))
#             sp_target = sp_target.masked_select(sp_mask.unsqueeze(-1))

            ap_loss = self.mse_loss(ap_output, ap_target)
#             print(sp_output.shape,sp_target.shape)
            sp_loss = self.mse_loss(sp_output, sp_target)
            sp_postnet_loss = self.mse_loss(sp_postnet_output, sp_target)

            return ap_loss, sp_loss, sp_postnet_loss, d_loss, p_loss, e_loss
        else:
            mel_output = mel_output.masked_select(mel_mask.unsqueeze(-1))
            mel_postnet_output = mel_postnet_output.masked_select(mel_mask.unsqueeze(-1))
            mel_target = mel_target.masked_select(mel_mask.unsqueeze(-1))

            mel_loss = self.mse_loss(mel_output, mel_target)
            mel_postnet_loss = self.mse_loss(mel_postnet_output, mel_target)

            return mel_loss, mel_postnet_loss, d_loss, p_loss, e_loss
