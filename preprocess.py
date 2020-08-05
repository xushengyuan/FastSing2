import os
from data import fastsing_dataset
import hparams as hp

def write_metadata(train,val, out_dir):
    with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        for m in train:
            f.write(m + '\n')
    with open(os.path.join(out_dir, 'val.txt'), 'w', encoding='utf-8') as f:
        for m in val:
            f.write(m + '\n')
            
def main():
    in_dir = hp.data_path
    out_dir = hp.preprocessed_path
    con_out_dir = os.path.join(out_dir, "condition")
    if not os.path.exists(con_out_dir):
        os.makedirs(con_out_dir, exist_ok=True)
    ap_out_dir = os.path.join(out_dir, "ap")
    if not os.path.exists(ap_out_dir):
        os.makedirs(ap_out_dir, exist_ok=True)
    sp_out_dir = os.path.join(out_dir, "sp")
    if not os.path.exists(sp_out_dir):
        os.makedirs(sp_out_dir, exist_ok=True)
    mel_out_dir = os.path.join(out_dir, "mel")
    if not os.path.exists(mel_out_dir):
        os.makedirs(mel_out_dir, exist_ok=True)
    ali_out_dir = os.path.join(out_dir, "alignment")
    if not os.path.exists(ali_out_dir):
        os.makedirs(ali_out_dir, exist_ok=True)
    f0_out_dir = os.path.join(out_dir, "f0")
    if not os.path.exists(f0_out_dir):
        os.makedirs(f0_out_dir, exist_ok=True)
    energy_out_dir = os.path.join(out_dir, "energy")
    if not os.path.exists(energy_out_dir):
        os.makedirs(energy_out_dir, exist_ok=True)

    train, val = fastsing_dataset.build_from_path(in_dir, out_dir)
    write_metadata(train, val, out_dir)
    
if __name__ == "__main__":
    main()
