import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import math
import os

import hparams as hp
import audio as Audio
from utils import pad_1D, pad_2D, process_meta
from text import text_to_sequence, sequence_to_text
from GST import GST
import h5py

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device='cpu'

class Dataset(Dataset):
    def __init__(self, filename="train.txt", sort=True):
        self.basename = process_meta(os.path.join(hp.preprocessed_path, filename))
        self.sort = sort

    def __len__(self):
        return len(self.basename)

    def __getitem__(self, idx):
        basename = self.basename[idx]
#         phone = np.array(text_to_sequence(self.text[idx], []))
        data=np.load(os.path.join(hp.preprocessed_path,"{}-{}.npz".format(hp.dataset, basename)))
    
        condition = data['condition'].T
        condition=np.clip(condition,1,1000)
        mel_refer = data['mel']
        ap_target = data['ap']
        sp_target = data['sp']
        
        D = data['duration']
        
        f0 = data['f0']
        f0=np.clip(f0,40,90)
        
        energy = data['energy']
        energy=np.clip(energy,-15,-5)
        energy=(energy+15)/10.0
        
#         assert D.sum()==f0.shape[0]==energy.shape[0]==ap_target.shape[0]==sp_target.shape[0]==mel_refer.shape[0]
        
        assert condition.shape[0]>0
        assert np.sum(D)>0
        assert sp_target.shape[0]>0
        assert not np.any(np.isnan(condition))
        assert not np.any(np.isnan(mel_refer))
        assert not np.any(np.isnan(ap_target))
        assert not np.any(np.isnan(sp_target))
        assert not np.any(np.isnan(D))
        assert not np.any(np.isnan(f0))
        assert not np.any(np.isnan(energy))
        
        
        norm_f0=np.zeros(f0.shape[0])
        for i in range(condition.shape[0]):
            for j in range(int(D[:i].sum()),min(int(D[:i].sum()+D[i]),f0.shape[0])):
                    norm_f0[j]=(condition[i][1])
        f0_norm=np.clip(norm_f0,40,90)
        D=np.clip(D,1,1000)
        if hp.vocoder=='WORLD':
            sample = {"id": basename,
                  "condition": condition,
                  "mel_refer":mel_refer,
                  "ap_target":ap_target,
                  "sp_target":sp_target,
                  "D": D,
                  "f0": f0,
                  "f0_norm": f0_norm,
                  "energy": energy}
        else:
            sample = {"id": basename,
                  "condition": condition,
                  "mel_refer":mell_refer,
                  "mel_target": mel_target,
                  "D": D,
                  "f0": f0,
                  "energy": energy}

        return sample

    def reprocess(self, batch, cut_list):
        ids = [batch[ind]["id"] for ind in cut_list]
        conditions = [batch[ind]["condition"] for ind in cut_list]
        mel_refers = [batch[ind]["mel_refer"] for ind in cut_list]
        if hp.vocoder=='WORLD':
            ap_targets = [batch[ind]["ap_target"] for ind in cut_list]
            sp_targets = [batch[ind]["sp_target"] for ind in cut_list]
        else:
            mel_targets = [batch[ind]["mel_target"] for ind in cut_list]
        Ds = [batch[ind]["D"] for ind in cut_list]
        f0s = [batch[ind]["f0"] for ind in cut_list]
        f0_norms = [batch[ind]["f0_norm"] for ind in cut_list]
        energies = [batch[ind]["energy"] for ind in cut_list]
        
        for condition, D, id_ in zip(conditions, Ds, ids):
            if len(condition) != len(D):
                print(condition, condition.shape, D, D.shape, id_)
                
        length_condition = np.array(list())
        for condition in conditions:
            length_condition = np.append(length_condition,condition.shape[0])

        length_mel = np.array(list())
        if hp.vocoder=='WORLD':
            for mel in sp_targets:
                length_mel = np.append(length_mel, mel.shape[0])
        else:
            for mel in mel_targets:
                length_mel = np.append(length_mel, mel.shape[0])
        
        conditions = pad_2D(conditions)
        Ds = pad_1D(Ds)
        mel_refers = pad_2D(mel_refers)
        if hp.vocoder=='WORLD':
            ap_targets = pad_2D(ap_targets)
            sp_targets = pad_2D(sp_targets)
#             print(ap_targets.shape,sp_targets.shape)
        else:
            mel_targets = pad_2D(mel_targets)
        f0s = pad_1D(f0s)
        f0s=np.clip(f0s,40,90)
        f0_norms = pad_1D(f0_norms)
        f0_norms=np.clip(f0_norms,40,90)
        energies = pad_1D(energies)
        log_Ds = np.log(Ds + hp.log_offset)

        if hp.vocoder=='WORLD':    
            out = {"id": ids,
               "condition": conditions,
               "mel_refer": mel_refers,
               "ap_target": ap_targets,
               "sp_target": sp_targets,
               "D": Ds,
               "log_D": log_Ds,
               "f0": f0s,
               "f0_norm": f0_norms,
               "energy": energies,
               "src_len": length_condition,
               "mel_len": length_mel}
        else:
            out = {"id": ids,
               "condition": conditions,
               "mel_refer": mel_refers,
               "mel_target": mel_targets,
               "D": Ds,
               "log_D": log_Ds,
               "f0": f0s,
               "energy": energies,
               "src_len": length_condition,
               "mel_len": length_mel}
        
        return out

    def collate_fn(self, batch):
        len_arr = np.array([d["condition"].shape[0] for d in batch])
        index_arr = np.argsort(-len_arr)
        batchsize = len(batch)
        real_batchsize = int(batchsize/4)

        cut_list = list()
        for i in range(4):
            if self.sort:
                cut_list.append(index_arr[i*real_batchsize:(i+1)*real_batchsize])
            else:
                cut_list.append(np.arange(i*real_batchsize, (i+1)*real_batchsize))
        
        output = list()
        for i in range(4):
            output.append(self.reprocess(batch, cut_list[i]))

        return output

if __name__ == "__main__":
    # Test
    dataset = Dataset('train.txt')
    training_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=dataset.collate_fn,
        drop_last=True, num_workers=0)
    total_step = hp.epochs * len(training_loader) * hp.batch_size

    cnt = 0
    for i, batchs in enumerate(training_loader):
        for j, data_of_batch in enumerate(batchs):
            sp_target = torch.from_numpy(
                data_of_batch["sp_target"]).float().to(device)
            D = torch.from_numpy(data_of_batch["D"]).int().to(device)
            if mel_target.shape[1] == D.sum().item():
                cnt += 1

    print(cnt, len(dataset))
