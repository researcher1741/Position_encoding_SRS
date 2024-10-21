from os.path import join
import os
from pathlib import Path

# ---- BASE PATHS  ----
import torch.nn as nn

BASE_PATH = Path(__file__).parents[1].__str__()

DATA_PATH = join(BASE_PATH, "Data")

# ########################    ITEMS    ###########################
BEAUTY_ITEMS = join(DATA_PATH, "Beauty_feat_cat.dat")
FASHION_ITEMS = join(DATA_PATH, "Fashion_imgs.dat")
SUBFASHION_ITEMS = join(DATA_PATH, "SubFashion_imgs.dat")
MEN_ITEMS = join(DATA_PATH, "Men_imgs.dat")
SUBMEN_ITEMS = join(DATA_PATH, "SubMen_imgs.dat")
SUBMEN2_ITEMS = join(DATA_PATH, "SubMen2_imgs.dat")
SUBMEN3_ITEMS = join(DATA_PATH, "SubMen3_imgs.dat")
GAMES_ITEMS = join(DATA_PATH, "Video_Games_feat.dat")
SUBGAMES_ITEMS = join(DATA_PATH, "Video_SubGames_feat.dat")
SUBGAMES2_ITEMS = join(DATA_PATH, "Video_SubGames_feat2.dat")
SUBGAMES3_ITEMS = join(DATA_PATH, "Video_SubGames_feat3.dat")
SUBGAMES4_ITEMS = join(DATA_PATH, "Video_SubGames_feat4.dat")

# ########################    TXT    ###########################
BEAUTY_TXT = join(DATA_PATH, "Beauty.txt")
FASHION_TXT = join(DATA_PATH, "Fashion.txt")
SUBFASHION_TXT = join(DATA_PATH, "SubFashion.txt")
MEN_TXT = join(DATA_PATH, "Men.txt")
SUBMEN_TXT = join(DATA_PATH, "SubMen.txt")
SUBMEN2_TXT = join(DATA_PATH, "SubMen2.txt")
SUBMEN3_TXT = join(DATA_PATH, "SubMen3.txt")
GAMES_TXT = join(DATA_PATH, "Video_Games.txt")
SUBGAMES_TXT = join(DATA_PATH, "Video_SubGames.txt")
SUBGAMES2_TXT = join(DATA_PATH, "Video_SubGames2.txt")
SUBGAMES3_TXT = join(DATA_PATH, "Video_SubGames3.txt")
SUBGAMES4_TXT = join(DATA_PATH, "Video_SubGames4.txt")

# ########################    CXT    ###########################
BEAUTY_CXT = join(DATA_PATH, "CXTDictSasRec_Beauty.dat")
FASHION_CXT = join(DATA_PATH, "CXTDictSasRec_Fashion.dat")
SUBFASHION_CXT = join(DATA_PATH, "CXTDictSasRec_SubFashion.dat")
MEN_CXT = join(DATA_PATH, "CXTDictSasRec_Men.dat")
SUBMEN_CXT = join(DATA_PATH, "CXTDictSasRec_SubMen.dat")
SUBMEN2_CXT = join(DATA_PATH, "CXTDictSasRec_SubMen2.dat")
SUBMEN3_CXT = join(DATA_PATH, "CXTDictSasRec_SubMen3.dat")
GAMES_CXT = join(DATA_PATH, "CXTDictSasRec_Games.dat")
SUBGAMES_CXT = join(DATA_PATH, "CXTDictSasRec_SubGames.dat")
SUBGAMES2_CXT = join(DATA_PATH, "CXTDictSasRec_SubGames2.dat")
SUBGAMES3_CXT = join(DATA_PATH, "CXTDictSasRec_SubGames3.dat")
SUBGAMES4_CXT = join(DATA_PATH, "CXTDictSasRec_SubGames4.dat")

# ########################    RESULTS    ###########################
RESULTS = join(DATA_PATH, "Results")
CHECK = join(DATA_PATH, "Checkpoints")

ACT2FN = {
    "relu": nn.ReLU(),
    "leakyrelu": nn.LeakyReLU(negative_slope=0.2),
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
}


class Args:
    """
    Args:
        hidden_units (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_encoder_blocks (`int`, *optional*, defaults to 3):
            Number of encoder blocks
        num_decoder_blocks (`int`, *optional*, defaults to 1):
            Number of decoder blocks
        num_heads (`int`, *optional*, defaults to 1):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.5):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.5):
            The dropout ratio for the attention probabilities.
        maxlen (`int`, *optional*, defaults to 75):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`BertModel`] or [`TFBertModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-8):
            The epsilon used by the layer normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        classifier_dropout (`float`, *optional*):
            The dropout ratio for the classification head.
    """

    def __init__(self, **kwargs):
        self.norm_type = 2.0
        self.max_norm = 1e-4
        # batch_size must be different of maxlen, otherwise the metrics fail
        # Saving parameters
        self.dataset = ''
        self.train_dir = 'default'
        # Training parameters
        self.batch_size = 256  # 128
        self.lr = 0.0001
        self.std = 0.01
        # Architecture for dataset
        self.maxlen = 75
        self.hidden_units = 90
        self.num_heads = 1  # num_heads = 1
        self.pad_token_id = 0  # pad_token_id
        self.num_epochs = int(16 * 100)
        self.dropout_rate = 0.5
        self.cxt_size = 6
        self.n_workers = 1
        self.top_k = 10  # [10]
        self.test_size = 10000  # 10000
        self.validation_point = int(1)  # 0.005  # integer means the amount of steps while float means percentage
        self.print_every_n_point = 1  # integer means the amount of steps while float means percentage
        self.exponential_print = True
        # EXPERIMENTS - Sampling
        self.last_items = False  # True
        self.reverse = True  # True
        self.only_finals = True  # True
        self.sampling_mode = False  # False
        # EXPERIMENTS - User
        self.add_users = False
        self.mask_user = self.add_users
        self.user_act = "silu"
        self.user_FF = True
        # EXPERIMENTS - Loss
        self.loss_type = "CE"
        # EXPERIMENTS - Encoding
        self.positional_encoding_type = ""  # "learnt", "absolute", "", "rotatory", "rope1"
        self.position_concatenation = False
        self.RMHA_encoder = False
        self.ROPE_encoder = False
        self.decoder_head = "masked"  # "RMHA", "dot"
        self.max_relative_position = 4
        # EXPERIMENTS - Architecture
        self.normalization_type = "ln"
        self.num_groups = 3
        self.residual_connection_encoder_FF = "mul"  # False in the code
        self.residual_connection_encoder = "mul"
        self.residual_connection_decoder = "mul"
        self.dropout_btn_MHA_FF = False  # False
        self.num_encoder_blocks = 3
        self.num_decoder_blocks = 1
        self.ln_in_AH_decoder = False
        self.ln_in_AH_encoder = True
        self.ln_in_Q_decoder = False
        self.ln_in_Q_encoder = True
        self.layer_norm_eps = 1e-8
        ####
        self.hidden_act = "leakyrelu"  # "swiglu2", "swiglu", "leakyrelu", "silu"
        self.hidden_act_out = "sigmoid"
        self.intercalate_act = True
        self.mask_before_FF_decoder = True
        self.PN = False
        self._add(**kwargs)

    def _add(self, **kwargs: object) -> object:
        for argument, value in kwargs.items():
            setattr(self, argument, value)
        self.embedding_d = self.hidden_units
        self.embedding_g = self.hidden_units * 5
        self.intermediate_size = self.hidden_units * 3
        self.hidden_dropout_prob = self.dropout_rate  # dropout_rate=0.5
        self.attention_probs_dropout_prob = self.dropout_rate  # dropout_rate=0.5
        self.saving = False


Beauty_Args = Args(**{"dataset": 'Beauty',
                      'batch_size': 256,
                      "norm_type": 2.0,
                      "max_norm": 1e-4,
                      'num_epochs': int(22 * 100),
                      })
Men_Args = Args(**{"dataset": 'Men',
                   'num_heads': 3,
                   'batch_size': 512,
                   'maxlen': 35,
                   'norm_type': 1e-2,
                   'max_norm': 1e-4,
                   'hidden_units': 390,
                   'lr': 6e-6,
                   'num_epochs': int(10 * 100),
                   'dropout_rate': 0.3,
                   'residual_connection_decoder': False})
SubMen_Args = Args(**{"dataset": 'SubMen',
                      'num_heads': 3,
                      'batch_size': 512,
                      'maxlen': 35,
                      'norm_type': 1e-2,
                      'max_norm': 1e-4,
                      'hidden_units': 390,
                      'lr': 6e-6,
                      'num_epochs': int(10 * 100),
                      'dropout_rate': 0.3,
                      'residual_connection_decoder': False})
SubMen2_Args = Args(**{"dataset": 'SubMen2',  # 10k users, 40k items
                       'num_heads': 3,
                       'batch_size': 512,
                       'maxlen': 35,
                       'norm_type': 1e-2,
                       'max_norm': 1e-4,
                       'hidden_units': 390,
                       'lr': 6e-6,
                       'num_epochs': int(10 * 100),
                       'dropout_rate': 0.3,
                       'residual_connection_decoder': False})
SubMen3_Args = Args(**{"dataset": 'SubMen3',  # 10k users, 80k items
                       'num_heads': 3,
                       'batch_size': 512,
                       'maxlen': 35,
                       'norm_type': 1e-2,
                       'max_norm': 1e-4,
                       'hidden_units': 390,
                       'lr': 6e-6,
                       'num_epochs': int(10 * 100),
                       'dropout_rate': 0.3,
                       'residual_connection_decoder': False})
Games_Args = Args(**{"dataset": 'Video_Games',
                     'num_encoder_blocks': 1,
                     'num_heads': 3,
                     'batch_size': 512,
                     'maxlen': 50,
                     'norm_type': 2.0,
                     'max_norm': 0.2,
                     'hidden_units': 90,
                     'lr': 1e-4,
                     'num_epochs': int(8 * 100),
                     'dropout_rate': 0.5,
                     'residual_connection_decoder': "mul",
                     'hidden_act': "silu"
                     })
SubGames_Args = Args(**{"dataset": 'Video_SubGames',
                        'num_encoder_blocks': 1,
                        'num_heads': 3,
                        'batch_size': 512,
                        'maxlen': 50,
                        'norm_type': 2.0,
                        'max_norm': 0.2,
                        'hidden_units': 90,
                        'lr': 1e-4,
                        'num_epochs': int(8 * 100),
                        'dropout_rate': 0.5,
                        'residual_connection_decoder': "mul",
                        'hidden_act': "silu"
                        })
SubGames2_Args = Args(**{"dataset": 'Video_SubGames2',
                         'num_encoder_blocks': 1,
                         'num_heads': 3,
                         'batch_size': 512,
                         'maxlen': 50,
                         'norm_type': 2.0,
                         'max_norm': 0.2,
                         'hidden_units': 90,
                         'lr': 1e-4,
                         'num_epochs': int(8 * 100),
                         'dropout_rate': 0.5,
                         'residual_connection_decoder': "mul",
                         'hidden_act': "silu"
                         })
SubGames3_Args = Args(**{"dataset": 'Video_SubGames3',
                         'num_encoder_blocks': 1,
                         'num_heads': 3,
                         'batch_size': 512,
                         'maxlen': 50,
                         'norm_type': 2.0,
                         'max_norm': 0.2,
                         'hidden_units': 90,
                         'lr': 1e-4,
                         'num_epochs': int(8 * 100),
                         'dropout_rate': 0.5,
                         'residual_connection_decoder': "mul",
                         'hidden_act': "silu"
                         })
SubGames4_Args = Args(**{"dataset": 'Video_SubGames4',
                         'num_encoder_blocks': 1,
                         'num_heads': 3,
                         'batch_size': 512,
                         'maxlen': 50,
                         'norm_type': 2.0,
                         'max_norm': 0.2,
                         'hidden_units': 90,
                         'lr': 1e-4,
                         'num_epochs': int(8 * 100),
                         'dropout_rate': 0.5,
                         'residual_connection_decoder': "mul",
                         'hidden_act': "silu"
                         })
Fashion_Args = Args(**{"dataset": 'Fashion',
                       'num_heads': 3,
                       'batch_size': 512,
                       'maxlen': 35,
                       'norm_type': 1e-4,
                       'max_norm': None,
                       'hidden_units': 390,
                       'lr': 1e-5,
                       'num_epochs': int(8 * 100),
                       'dropout_rate': 0.3,
                       'residual_connection_decoder': False})
SubFashion2_Args = Args(**{"dataset": 'SubFashion2',
                           'num_heads': 3,
                           'batch_size': 512,
                           'maxlen': 35,
                           'norm_type': 1e-4,
                           'max_norm': None,
                           'hidden_units': 90,
                           'lr': 1e-5,
                           'num_epochs': int(8 * 100),
                           'dropout_rate': 0.3,
                           'residual_connection_decoder': False})


def create_tree():
    list_of_folder = [RESULTS, CHECK]
    for folder in list_of_folder:
        if not os.path.exists(folder):
            os.makedirs(folder)


create_tree()

STATE_DICT_KEY = 'model_state_dict'
OPTIMIZER_STATE_DICT_KEY = 'optimizer_state_dict'
