import torch
from datetime import datetime

EPOCHS = 10000
STEPS = 100000
SAVE_STEPS = [1000, 10000, 25000, 50000, 100000]
BS = 64 #128 # simclr: 512 # 64
SOURCE_SAMPLE_RATE= 1000
TARGET_SAMPLE_RATE = 256
SAMPLE_PER_LABEL = TARGET_SAMPLE_RATE // 2
TIME_SEC = 10
SIG_LEN = TARGET_SAMPLE_RATE * TIME_SEC
PADDING = False # padding input size
CH_IDX = [0] # [0, 1, 2, 3, 4]
N_CHANNEL = len(CH_IDX)

CLASS_WEIGHT = [0.5417898193760262, 1.138543823326432, 3.623833058758924]
# [0.5600985221674877, 1.1770186335403727, 3.746293245469522, 10.197309417040358] # None
CLASS_NAMES = ['N', 'O', 'A'] # ['N', 'O', 'A', '~'] # sinus, other, AF, noise
NUM_CLS = len(CLASS_NAMES)
USE_CLASS_WEIGHT = True
BALANCE_CLASS = True # blance class samples to majority
FINE_TUNE = True # load pretrained
FREEZE = False
AUG = True
TRAIN_SIZE = 100 # in percent [1, 10, 25, 50]
print('fine tune: ', FINE_TUNE)
print('Freeze encoder: ', FREEZE)
print('Augmentation', AUG)
print('Class weight: ', USE_CLASS_WEIGHT)
print('Banlanc class: ', BALANCE_CLASS)
# File Path
FOLDER_PATH = "../../../Documents/亞東林/data/signal"
time_str = datetime.now().strftime("%Y%m%d_%H%M")
MODEL_PATH = f"../../../Documents/ntuecg/model_physio/model_cls_{time_str}.pth"
default_root_dir = "../../../Documents/ntuecg/physiosslmodel/"
TENSORBOARD_PATH = '../../../Documents/ntuecg/physiolog/'
# Model
MODELTYPE = "convnext" # convnext, cnn, transformer
print('model : ', MODELTYPE)
if MODELTYPE == "transformer":
    patch_size = 40
    dim = 512
    depth = 6
    heads = 8
    mlp_dim = 512
    dropout = 0.1
    emb_dropout = 0.1
    pool='cls'
elif MODELTYPE == "cnn":
    in_channels=N_CHANNEL
    base_filters=64
    kernel_size=16 
    stride=2 
    n_block=16 
    downsample_gap=2
    increasefilter_gap=4
    groups=32
    verbose=False
elif MODELTYPE == "convnext":
    depths=[3,3,9,3]# [3, 3, 13, 3], best: [3,3,3,3]
    dims=[32, 64, 128, 256] #[96, 192, 384, 768]
    base_kernel_size=13 # raw:4 best:7
    kernel_size=7
    # attention
    se = True
    cbam = False
    spatial_attention = True
# tiny: ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768])
# small: ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768])
# base: ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024])
# large: ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536])
# xlarge: ConvNeXt(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048])
else:
    raise AssertionError('wrong model type')

# SSL
SSLTYPE = "simclr" # 'scratch' "simsiam" "byol" "simclr" "barlowtwins"
print("SSL : ", SSLTYPE)
if SSLTYPE == "simclr":
    dim = 512 if MODELTYPE != "convnext" else dims[-1]
elif SSLTYPE == "simsiam":
    dim = 512 if MODELTYPE != "convnext" else dims[-1]
elif SSLTYPE == "byol":
    projector_hidden_size=4096
    projector_out_dim=256
elif SSLTYPE == 'barlowtwins':
    dim = 512 if MODELTYPE != "convnext" else dims[-1]
    projector = "-".join(["8192"]*3) #8192, 4096, 2048 "2048-2048-2048"# "8192-8192-8192"
    lambd = 0.0051
    weight_decay = 1e-6 # LARS opt
    warmup_epochs = 10
elif SSLTYPE == 'scratch':
    print('scratch')
else:
    raise AssertionError("wrong ssl type")
    
# Dataset
ptb = True
        
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device: ", device)

MODEL_NAME = f"encoder-{SSLTYPE}-{MODELTYPE}.pth"
VERBOSE = True

# LR_COSINE = True
DROP_OUT = 0.
