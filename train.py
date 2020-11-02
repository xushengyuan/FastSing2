import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import argparse
import os
import time

from fastspeech2 import FastSpeech2
from loss import FastSpeech2Loss
from dataset import Dataset
from optimizer import ScheduledOptim
from evaluate import evaluate
import hparams as hp
import utils
import audio as Audio
import soundfile as sf
from prefetch_generator import BackgroundGenerator
import matplotlib.pyplot as plt

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


    
def main(args):
    torch.manual_seed(0)

#    torch.distributed.init_process_group(backend="nccl",init_method="tcp://localhost:14421",rank=args.local_rank,world_size=4)
 #   local_rank = torch.distributed.get_rank()
 #   torch.cuda.set_device(local_rank)
  #  device = torch.device("cuda", local_rank)
    # Get device
    device = torch.device('cuda'if torch.cuda.is_available()else 'cpu')
    #device = 'cpu'
    
    # Get dataset
    dataset = Dataset("train.txt") 
    loader = DataLoaderX(dataset, batch_size=hp.batch_size*4, 
        collate_fn=dataset.collate_fn,shuffle=True, drop_last=True, num_workers=8)

    # Define model
    
#    model = FastSpeech2().to(device)
 #   model = torch.nn.parallel.DistributedDataParallel(model,
#device_ids=[local_rank],
#output_device=local_rank)
#     assert model is not None
    model = nn.DataParallel(FastSpeech2()).to(device)
#    model = FastSpeech2().to(device)
#
#    print("Model Has Been Defined")
    num_param = utils.get_param_num(model)
    print('Number of FastSpeech2 Parameters:', num_param)

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), betas=hp.betas, eps=hp.eps, weight_decay = hp.weight_decay)
    scheduled_optim = ScheduledOptim(optimizer, hp.decoder_hidden, hp.n_warm_up_step, args.restore_step)
    Loss = FastSpeech2Loss().to(device) 
    print("Optimizer and Loss Function Defined.")

    # Load checkpoint if exists
    checkpoint_path = os.path.join(hp.checkpoint_path)
    try:
        path=os.path.join(
            checkpoint_path, 'checkpoint_{}.pth.tar'.format(args.restore_step))
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("\n---Model Restored at Step {}---\n".format(args.restore_step))
        last_ckpt=args.restore_step
    except:
        if args.restore_step!=0:
            raise
        print("\n---Start New Training---\n")
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

    # Load vocoder
    if hp.vocoder == 'melgan':
        melgan = utils.get_melgan()
        melgan.to(device)
    elif hp.vocoder == 'waveglow':
        waveglow = utils.get_waveglow()
        waveglow.to(device)

    # Init logger
    log_path = hp.log_path
    if not os.path.exists(log_path):
        os.makedirs(log_path)
        
    current_time = time.strftime("%Y-%m-%dT%H:%M", time.localtime())
    
    train_logger =  SummaryWriter(log_dir='log/'+current_time)
    os.system("cp ./hparams.py "+'log/'+current_time+"/hparams.py")
    # Init synthesis directory
    synth_path = hp.synth_path
    if not os.path.exists(synth_path):
        os.makedirs(synth_path)

    # Define Some Information
    Time = np.array([])
    Start = time.perf_counter()
    current_step0=args.restore_step
    # Training
    assert model is not None

#     model = model.train()
    for epoch in range(hp.epochs):
        # Get Training Loader
        total_step = hp.epochs * len(loader) * hp.batch_size

        for i, batchs in enumerate(loader):
            for j, data_of_batch in enumerate(batchs):
                start_time = time.perf_counter()

                current_step = i * len(batchs) + j + args.restore_step + \
                    epoch * len(loader)*len(batchs) + 1
                # Get Data
                condition = torch.from_numpy(data_of_batch["condition"]).long().to(device)
                mel_refer = torch.from_numpy(data_of_batch["mel_refer"]).float().to(device)
                if hp.vocoder=='WORLD':
                    ap_target = torch.from_numpy(data_of_batch["ap_target"]).float().to(device)
                    sp_target = torch.from_numpy(data_of_batch["sp_target"]).float().to(device)
                else:
                    mel_target = torch.from_numpy(data_of_batch["mel_target"]).float().to(device)
                D = torch.from_numpy(data_of_batch["D"]).long().to(device)
                log_D = torch.from_numpy(data_of_batch["log_D"]).float().to(device)
                #print(D,log_D)
                f0 = torch.from_numpy(data_of_batch["f0"]).float().to(device)
                f0_norm = torch.from_numpy(data_of_batch["f0_norm"]).float().to(device)
                energy = torch.from_numpy(data_of_batch["energy"]).float().to(device)
                src_len = torch.from_numpy(data_of_batch["src_len"]).long().to(device)
                mel_len = torch.from_numpy(data_of_batch["mel_len"]).long().to(device)
                max_src_len = np.max(data_of_batch["src_len"]).astype(np.int32)
                max_mel_len = np.max(data_of_batch["mel_len"]).astype(np.int32)
                
                if hp.vocoder=='WORLD':
                    #print(torch.mean(condition),torch.mean(mel_refer), torch.mean(src_len), torch.mean(mel_len), torch.mean(D), torch.mean(f0), torch.mean(energy), torch.mean(max_src_len), torch.mean(max_mel_len))
                    assert model is not None

                    ap_output, sp_output, sp_postnet_output, log_duration_output, f0_output,energy_output, src_mask, ap_mask,sp_mask ,variance_adaptor_output,decoder_output,encoder_output= model(
                    condition, src_len, mel_len, D, f0, f0_norm, energy, max_src_len, max_mel_len)
                
#                     print(torch.mean(ap_output), torch.mean(sp_output), torch.mean(sp_postnet_output), torch.mean(log_duration_output), torch.mean(f0_output),torch.mean(energy_output) ,torch.mean(variance_adaptor_output),torch.mean(decoder_output),torch.mean(encoder_output))
                    ap_loss, sp_loss, sp_postnet_loss, d_loss, f_loss, e_loss = Loss(
                        log_duration_output, D, f0_output, f0, energy_output, energy, ap_output=ap_output, 
                        sp_output=sp_output, sp_postnet_output=sp_postnet_output, ap_target=ap_target, 
                        sp_target=sp_target,src_mask=src_mask, ap_mask=ap_mask,sp_mask=sp_mask)
                    total_loss =ap_loss + sp_loss + sp_postnet_loss + d_loss + f_loss + e_loss
                    #print(energy,energy_output)
#                     print(ap_loss , sp_loss , sp_postnet_loss , d_loss , f_loss, e_loss)
   
                    if torch.isnan(total_loss):

                         path=os.path.join(
            checkpoint_path, 'checkpoint_{}.pth.tar'.format(last_ckpt))
                         checkpoint = torch.load(path)
                         model.load_state_dict(checkpoint['model'])
                         optimizer.load_state_dict(checkpoint['optimizer'])
                         print("\n---Model Restored at Step {}---\n".format(last_ckpt))
                else:
                    mel_output, mel_postnet_output, log_duration_output, f0_output,energy_output, src_mask, mel_mask, _ = model(
                    condition,mel_refer, src_len, mel_len, D, f0, energy, max_src_len, max_mel_len)
                    
                    mel_loss, mel_postnet_loss, d_loss, f_loss, e_loss = Loss(
                        log_duration_output, log_D, f0_output, f0, energy_output, energy, mel_output=mel_output,
                        mel_postnet_output=mel_postnet_output, mel_target=mel_target, src_mask=~src_mask, mel_mask=~mel_mask)
                    total_loss = mel_loss + mel_postnet_loss + d_loss + f_loss+ e_loss
                     
               

                # Logger
                t_l = total_loss.item()
                if hp.vocoder=='WORLD':
                    ap_l = ap_loss.item()
                    sp_l = sp_loss.item()
                    sp_p_l = sp_postnet_loss.item()
                else:
                    m_l = mel_loss.item()
                    m_p_l = mel_postnet_loss.item()
                d_l = d_loss.item()
                f_l = f_loss.item()
                e_l = e_loss.item()
#                 with open(os.path.join(log_path, "total_loss.txt"), "a") as f_total_loss:
#                     f_total_loss.write(str(t_l)+"\n")
#                 with open(os.path.join(log_path, "mel_loss.txt"), "a") as f_mel_loss:
#                     f_mel_loss.write(str(m_l)+"\n")
#                 with open(os.path.join(log_path, "mel_postnet_loss.txt"), "a") as f_mel_postnet_loss:
#                     f_mel_postnet_loss.write(str(m_p_l)+"\n")
#                 with open(os.path.join(log_path, "duration_loss.txt"), "a") as f_d_loss:
#                     f_d_loss.write(str(d_l)+"\n")
#                 with open(os.path.join(log_path, "f0_loss.txt"), "a") as f_f_loss:
#                     f_f_loss.write(str(f_l)+"\n")
#                 with open(os.path.join(log_path, "energy_loss.txt"), "a") as f_e_loss:
#                     f_e_loss.write(str(e_l)+"\n")
                 
                # Backward
                total_loss = total_loss / hp.acc_steps
                if not torch.isnan(total_loss):
                    total_loss.backward()
                if current_step % hp.acc_steps != 0:
                    continue

                # Clipping gradients to avoid gradient explosion
                nn.utils.clip_grad_norm_(model.parameters(), hp.grad_clip_thresh)

                # Update weights
                scheduled_optim.step_and_update_lr()
                scheduled_optim.zero_grad()
                
                end_time = time.perf_counter()
                Time = np.append(Time, end_time - start_time)
                if len(Time) == hp.clear_Time:
                    temp_value = np.mean(Time)
                    Time = np.delete(
                        Time, [i for i in range(len(Time))], axis=None)
                    Time = np.append(Time, temp_value)
                
                # Print
                if  current_step % hp.log_step == 0:
                    Now = time.perf_counter()

                    str1 = "Epoch[{}/{}],Step[{}/{}]:".format(
                        epoch+1, hp.epochs, current_step, total_step)
                    if hp.vocoder=='WORLD':
                        str2 = "Loss:{:.4f},ap:{:.4f},sp:{:.4f},spPN:{:.4f},Dur:{:.4f},F0:{:.4f},Energy:{:.4f};".format(t_l, ap_l, sp_l, sp_p_l, d_l, f_l, e_l)
                    else:
                        str2 = "Loss:{:.4f},Mel:{:.4f},MelPN:{:.4f},Dur:{:.4f},F0:{:.4f},Energy:{:.4f};".format(
                        t_l, m_l, m_p_l, d_l, f_l, e_l)
                    str3 = "LR:{:.6f}T:{:.2f}sTa:{:.2f}s".format(
                       scheduled_optim.get_learning_rate(), end_time - start_time,np.mean(Time) )

                    print("" + str1+str2+str3+'')
                    
#                     with open(os.path.join(log_path, "log.txt"), "a") as f_log:
#                         f_log.write(str1 + "\n")
#                         f_log.write(str2 + "\n")
#                         f_log.write(str3 + "\n")
#                         f_log.write("\n")
#                 if local_rank==0:
                train_logger.add_scalar('Loss/total_loss', t_l, current_step)
                if hp.vocoder=='WORLD':
                    train_logger.add_scalar('Loss/ap_loss', ap_l, current_step)
                    train_logger.add_scalar('Loss/sp_loss', sp_l, current_step)
                    train_logger.add_scalar('Loss/sp_postnet_loss', sp_p_l, current_step)
                else:
                    train_logger.add_scalar('Loss/mel_loss', m_l, current_step)
                    train_logger.add_scalar('Loss/mel_postnet_loss', m_p_l, current_step)
                train_logger.add_scalar('Loss/duration_loss', d_l, current_step)
                train_logger.add_scalar('Loss/F0_loss', f_l, current_step)
                train_logger.add_scalar('Loss/energy_loss', e_l, current_step)
                train_logger.add_scalar('learning_rate',scheduled_optim.get_learning_rate() , current_step)
                
                if current_step % hp.save_step == 0 or current_step==20:
                    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(
                    )}, os.path.join(checkpoint_path, 'checkpoint_{}.pth.tar'.format(current_step)))
                    print("save model at step {} ...".format(current_step))
                    last_ckpt=current_step

                if current_step % hp.synth_step == 0 or current_step==24010:
                    length = mel_len[0].item()
                    
                    if hp.vocoder=='WORLD':
                        ap_target_torch = ap_target[0, :length].detach().unsqueeze(0).transpose(1, 2)
                        ap_torch = ap_output[0, :length].detach().unsqueeze(0).transpose(1, 2)
                        sp_target_torch = sp_target[0, :length].detach().unsqueeze(0).transpose(1, 2)
                        sp_torch = sp_output[0, :length].detach().unsqueeze(0).transpose(1, 2)
                        sp_postnet_torch = sp_postnet_output[0, :length].detach().unsqueeze(0).transpose(1, 2)
                    else:
                        mel_target_torch = mel_target[0, :length].detach().unsqueeze(0).transpose(1, 2)
                        mel_target = mel_target[0, :length].detach().cpu().transpose(0, 1)
                        mel_torch = mel_output[0, :length].detach().unsqueeze(0).transpose(1, 2)
                        mel = mel_output[0, :length].detach().cpu().transpose(0, 1)
                        mel_postnet_torch = mel_postnet_output[0, :length].detach().unsqueeze(0).transpose(1, 2)
                        mel_postnet = mel_postnet_output[0, :length].detach().cpu().transpose(0, 1)
#                     Audio.tools.inv_mel_spec(mel, os.path.join(synth_path, "step_{}_griffin_lim.wav".format(current_step)))
#                     Audio.tools.inv_mel_spec(mel_postnet, os.path.join(synth_path, "step_{}_postnet_griffin_lim.wav".format(current_step)))

                    f0 = f0[0, :length].detach().cpu().numpy()
                    energy = energy[0, :length].detach().cpu().numpy()
                    f0_output = f0_output[0, :length].detach().cpu().numpy()
                    f0_norm = f0_norm[0, :length].detach().cpu().numpy()
                    energy_output = energy_output[0, :length].detach().cpu().numpy()

                    if hp.vocoder == 'melgan':
                        utils.melgan_infer(mel_torch, melgan, os.path.join(hp.synth_path, 'step_{}_{}.wav'.format(current_step, hp.vocoder)))
                        utils.melgan_infer(mel_postnet_torch, melgan, os.path.join(hp.synth_path, 'step_{}_postnet_{}.wav'.format(current_step, hp.vocoder)))
                        utils.melgan_infer(mel_target_torch, melgan, os.path.join(hp.synth_path, 'step_{}_ground-truth_{}.wav'.format(current_step, hp.vocoder)))
                    elif hp.vocoder == 'waveglow':
                        utils.waveglow_infer(mel_torch, waveglow, os.path.join(hp.synth_path, 'step_{}_{}.wav'.format(current_step, hp.vocoder)))
                        utils.waveglow_infer(mel_postnet_torch, waveglow, os.path.join(hp.synth_path, 'step_{}_postnet_{}.wav'.format(current_step, hp.vocoder)))
                        utils.waveglow_infer(mel_target_torch, waveglow, os.path.join(hp.synth_path, 'step_{}_ground-truth_{}.wav'.format(current_step, hp.vocoder)))
                    elif hp.vocoder=='WORLD':
                        #     ap=np.swapaxes(ap,0,1)
                        #     sp=np.swapaxes(sp,0,1)
                        wav=utils.world_infer(np.swapaxes(ap_torch[0].cpu().numpy(),0,1),
                                              np.swapaxes(sp_postnet_torch[0].cpu().numpy(),0,1),f0_output)
                        sf.write(os.path.join(hp.synth_path, 'step_{}_postnet_{}.wav'.format(current_step, hp.vocoder)),wav,32000)
                        wav=utils.world_infer(np.swapaxes(ap_target_torch[0].cpu().numpy(),0,1),
                                              np.swapaxes(sp_target_torch[0].cpu().numpy(),0,1),f0)
                        sf.write(os.path.join(hp.synth_path, 'step_{}_ground-truth_{}.wav'.format(current_step, hp.vocoder)),wav,32000)
                    
                    utils.plot_data([(sp_postnet_torch[0].cpu().numpy(), f0_output,f0_norm, energy_output), (sp_target_torch[0].cpu().numpy(), f0,f0_norm, energy)], 
                        ['Synthetized Spectrogram', 'Ground-Truth Spectrogram'], filename=os.path.join(synth_path, 'step_{}.png'.format(current_step)))
                    
                    plt.matshow(sp_postnet_torch[0].cpu().numpy())
                    plt.savefig(os.path.join(synth_path, 'sp_postnet_{}.png'.format(current_step)))
                    plt.matshow(ap_torch[0].cpu().numpy())
                    plt.savefig(os.path.join(synth_path, 'ap_{}.png'.format(current_step)))
                    plt.matshow(variance_adaptor_output[0].detach().cpu().numpy())
#                     plt.savefig(os.path.join(synth_path, 'va_{}.png'.format(current_step)))
#                     plt.matshow(decoder_output[0].detach().cpu().numpy())
#                     plt.savefig(os.path.join(synth_path, 'encoder_{}.png'.format(current_step)))

                    plt.cla()
                    fout=open(os.path.join(synth_path, 'D_{}.txt'.format(current_step)),'w')
                    fout.write(str(log_duration_output[0].detach().cpu().numpy().astype(np.int32))+'\n')
                    fout.write(str(D[0].detach().cpu().numpy())+'\n')
                    fout.write(str(condition[0,:,2].detach().cpu().numpy())+'\n')
                    fout.close()
#                 if current_step % hp.eval_step == 0 or current_step==20:
#                     model.eval()
#                     with torch.no_grad():
                        
#                         if hp.vocoder=='WORLD':
#                             d_l, f_l, e_l, ap_l, sp_l, sp_p_l = evaluate(model, current_step)
#                             t_l = d_l + f_l + e_l + ap_l + sp_l + sp_p_l

#                             train_logger.add_scalar('valLoss/total_loss', t_l, current_step)
#                             train_logger.add_scalar('valLoss/ap_loss', ap_l, current_step)
#                             train_logger.add_scalar('valLoss/sp_loss', sp_l, current_step)
#                             train_logger.add_scalar('valLoss/sp_postnet_loss', sp_p_l, current_step)
#                             train_logger.add_scalar('valLoss/duration_loss', d_l, current_step)
#                             train_logger.add_scalar('valLoss/F0_loss', f_l, current_step)
#                             train_logger.add_scalar('valLoss/energy_loss', e_l, current_step)
#                         else:
#                             d_l, f_l, e_l, m_l, m_p_l = evaluate(model, current_step)
#                             t_l = d_l + f_l + e_l + m_l + m_p_l

#                             train_logger.add_scalar('valLoss/total_loss', t_l, current_step)
#                             train_logger.add_scalar('valLoss/mel_loss', m_l, current_step)
#                             train_logger.add_scalar('valLoss/mel_postnet_loss', m_p_l, current_step)
#                             train_logger.add_scalar('valLoss/duration_loss', d_l, current_step)
#                             train_logger.add_scalar('valLoss/F0_loss', f_l, current_step)
#                             train_logger.add_scalar('valLoss/energy_loss', e_l, current_step)

#                     model.train()
                    
                

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_step', type=int, default=0)
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    main(args)
