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
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    

    train, val = fastsing_dataset.build_from_path(in_dir, out_dir)
    write_metadata(train, val, out_dir)
    
if __name__ == "__main__":
    main()
