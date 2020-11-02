import os

# Dataset
dataset = "fastsing_dataset"
data_path = "../mu_yao"
#dataset = "Blizzard2013"
#data_path = "./Blizzard-2013/train/segmented/"


# Text
text_cleaners = ['english_cleaners']


# Audio and mel
### for LJSpeech ###
sampling_rate = 32000
filter_length = 1024
hop_length = 256
win_length = 1024
### for Blizzard2013 ###
#sampling_rate = 16000
#filter_length = 800
#hop_length = 200
#win_length = 800

max_wav_value = 32768.0
n_mel_channels = 128
n_ap_channels = 32
n_sp_channels = 128
mel_fmin = 75.0
mel_fmax = 8000.0

vocoder = 'WORLD'




n_condition=3

n_src_vocab=1024
# FastSpeech 2
encoder_layer = 4
encoder_head = 2
encoder_hidden = 256
decoder_layer = 4
decoder_head = 2
decoder_hidden = 256
fft_conv1d_filter_size = 1024
fft_conv1d_kernel_size = (9, 1)
encoder_dropout = 0.2
decoder_dropout = 0.2


variance_predictor_dropout = 0.2
variance_predictor_layer = 4
variance_predictor_head = 2
variance_predictor_hidden = 256
length_predictor_dropout = 0.2
length_predictor_layer = 4
length_predictor_head = 2
length_predictor_hidden = 256

max_seq_len = 4096


# Quantization for F0 and energy
### for LJSpeech ###
f0_min = 40.0
f0_max = 90.0
energy_min = 0
energy_max = 1
### for Blizzard2013 ###
#f0_min = 71.0
#f0_max = 786.7
#energy_min = 21.23
#energy_max = 101.02

n_bins = 128


# Checkpoints and synthesis path
preprocessed_path = os.path.join("../mu_yao/preprocessed/", dataset)
checkpoint_path = os.path.join("./ckpt/", dataset)
synth_path = os.path.join("./synth/", dataset)
eval_path = os.path.join("./eval/", dataset)
log_path = os.path.join("./log/", dataset)
test_path = "./results"

val_rate=0.00001


# Optimizer
batch_size = 16	

epochs = 1000
learning_rate=5e-5
learning_rate_ratio=1/8
n_warm_up_step = 4000
grad_clip_thresh = 1.0
acc_steps = 1

betas = (0.9, 0.98)
eps = 1e-9
weight_decay = 0.




# Log-scaled duration
log_offset = 1.


# Save, log and synthesis
save_step = 2000
synth_step = 1000
eval_step = 10000
eval_size = 256
log_step = 1
clear_Time = 20


E = 256

# reference encoder
ref_enc_n_mels= 256
ref_enc_filters = [32, 32, 64, 64, 128, 128, 256, 256]
ref_enc_size = [3, 3]
ref_enc_strides = [2, 2]
ref_enc_pad = [1, 1]
ref_enc_gru_size = E // 2

# style token layer
token_num = 10
# token_emb_size = 256
num_heads = 8
# multihead_attn_num_unit = 256
# style_att_type = 'm lp_attention'
# attn_normalize = True

K = 16
decoder_K = 8
embedded_size = E
dropout_p = 0.5
num_banks = 15
num_highways = 4
