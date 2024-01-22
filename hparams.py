import torch
import numpy as np
import random

def seed_everywhere(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic=True

class HParams:
    def __init__(self):
        self.USE_CPU = False
        self.USE_CUDA = True

        self.MODEL_TYPE = "s2s"
        self.N_EPOCHS = 48
        self.LEARNING_RATE = 0.0003
        self.BATCH_SIZE = 8
        self.HIDDEN_SIZE = 100
        self.DROPOUT = 0.3
        self.RNN_LAYERS = 2
        self.DECODER_LEARNING_RATIO = 5.0
        self.MAX_GRAD_NORM = 1
        self.MAX_POST_LEN = 100
        self.MAX_QUES_LEN = 20
        self.MAX_KWD = 5000
        self.ATTN_TYPE = 'dot'
        self.DECODER_CONDITION_TYPE = 'replace'  # or 'none', 'concat'

        # kwd predictor
        self.FREEZE_KWD_MODEL = False
        self.KWD_MODEL_LAYERS = 1
        self.PATIENCE = 4
        self.NEG_KWD_PER = 30
        self.MIN_NEG_KWD = self.NEG_KWD_PER
        self.KWD_PREDICTOR_TYPE = 'cnn'             # or gru
        self.NO_NEG_SAMPLE = False

        # kwd bridge
        self.BRIDGE_NORM_TYPE = 'dropout'           # or layer_norm, batch_norm, sigmoid, none
        self.HARD_KWD_BRIDGE = False

        # decode
        self.BEAM_SIZE = 6
        self.BLOCK_NGRAM_REPEAT = 2
        self.AVOID_REPEAT_WINDOW = 2  # avoid 1-gram repeat with the previous 2
        self.DECODE_USE_KWD_LABEL = False
        self.CLUSTER_KWD = False
        self.KWD_CLUSTERS = 2
        self.SHOW_TOP_KWD = 20
        self.THRESHOLD = -1.0
        self.SAMPLE_KWD = 4
        self.SAMPLE_TOP_K = 6
        self.SAMPLE_TOP_P = 0.9
        self.SAMPLE_DECODE_WORD = False   # currently hard-code top-20, top-0.9 sampling, BEAM_SIZE seqs
        self.USER_FILTER = False
        self.SAVE_EPOCH_INTERVAL = 4
        self.SEED = 77
        seed_everywhere(self.SEED)


        # diverse beam search
        self.DIVERSE_BEAM = False
        self.DIVERSE_GROUP = 3
        self.DIVERSE_LAMBDA = 0.4

        # not useful tricks
        # end2end kwd model training
        self.KWD_LOSS_RATIO = 1.0
        self.UPDATE_WD_EMB = False
        self.SCHEDULED_SAMPLE = False
        self.MIN_TF_RATIO = 0.2
        self.BALANCE_KWD_CLASS = False

        # kwd bridge
        self.WITH_MEMORY = False
        self.MEMORY_HOPS = 2
        self.NO_ENCODER_BRIDGE = False
        self.NO_DECODER_BRIDGE = False

        # personalize
        self.MAX_PROMPT_LEN = 5
