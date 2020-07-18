import argparse
import os
import torch


class Param:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="")

        # General
        self.parser.add_argument('--iters', type=int, default=100000)
        self.parser.add_argument('--name', type=str, default='default')
        self.parser.add_argument('--train', type=str, default='speaker')

        # Data preparation
        self.parser.add_argument('--maxInput', type=int, default=80, help="max input instruction")
        self.parser.add_argument('--maxDecode', type=int, default=120, help="max input instruction")
        self.parser.add_argument('--maxAction', type=int, default=35, help='Max Action sequence')
        self.parser.add_argument('--per_gpu_train_batch_size', dest='batchSize', type=int, default=64)
        self.parser.add_argument('--ignoreid', type=int, default=-100)
        self.parser.add_argument('--feature_size', type=int, default=2048)
        self.parser.add_argument("--loadOptim",action="store_const", default=False, const=True)

        # Load the model from
        self.parser.add_argument("--speaker", default=None)
        self.parser.add_argument("--listener", default=None)
        self.parser.add_argument("--load", type=str, default=None)

        # More Paths from
        self.parser.add_argument("--aug", default=None)

        # Listener Model Config
        self.parser.add_argument("--zeroInit", dest='zero_init', action='store_const', default=False, const=True)
        self.parser.add_argument("--mlWeight", dest='ml_weight', type=float, default=0.05)
        self.parser.add_argument("--teacherWeight", dest='teacher_weight', type=float, default=1.)
        self.parser.add_argument("--accumulateGrad", dest='accumulate_grad', action='store_const', default=False, const=True)
        self.parser.add_argument("--features", type=str, default='imagenet')

        # Env Dropout Param
        self.parser.add_argument('--featdropout', type=float, default=0.3)

        # SSL configuration
        self.parser.add_argument("--selfTrain", dest='self_train', action='store_const', default=False, const=True)

        # Submision configuration
        self.parser.add_argument("--candidates", type=int, default=1)
        self.parser.add_argument("--paramSearch", dest='param_search', action='store_const', default=False, const=True)
        self.parser.add_argument("--submit", action='store_const', default=False, const=True)
        self.parser.add_argument("--beam", action="store_const", default=False, const=True)
        self.parser.add_argument("--alpha", type=float, default=0.5)

        # Training Configurations
        self.parser.add_argument('--optim', type=str, default='rms')    # rms, adam
        self.parser.add_argument('--lr', type=float, default=0.0001, help="The learning rate")
        self.parser.add_argument('--decay', dest='weight_decay', type=float, default=0.)
        self.parser.add_argument('--dropout', type=float, default=0.5)
        self.parser.add_argument('--feedback', type=str, default='sample',
                            help='How to choose next position, one of ``teacher``, ``sample`` and ``argmax``')
        self.parser.add_argument('--teacher', type=str, default='final',
                            help="How to get supervision. one of ``next`` and ``final`` ")
        self.parser.add_argument('--epsilon', type=float, default=0.1)

        # Model hyper params:
        self.parser.add_argument('--rnnDim', dest="rnn_dim", type=int, default=512)
        self.parser.add_argument('--wemb', type=int, default=256)
        self.parser.add_argument('--aemb', type=int, default=64)
        self.parser.add_argument('--proj', type=int, default=512)
        self.parser.add_argument("--fast", dest="fast_train", action="store_const", default=False, const=True)
        self.parser.add_argument("--valid", action="store_const", default=False, const=True)
        self.parser.add_argument("--candidate", dest="candidate_mask",
                                 action="store_const", default=False, const=True)

        self.parser.add_argument("--bidir", type=bool, default=True)    # This is not full option
        self.parser.add_argument("--encode", type=str, default="word")  # sub, word, sub_ctx
        self.parser.add_argument("--subout", dest="sub_out", type=str, default="tanh")  # tanh, max
        self.parser.add_argument("--attn", type=str, default="soft")    # soft, mono, shift, dis_shift

        self.parser.add_argument("--angleFeatSize", dest="angle_feat_size", type=int, default=4)

        # A2C
        self.parser.add_argument("--gamma", default=0.9, type=float)
        self.parser.add_argument("--normalize", dest="normalize_loss", default="total", type=str, help='batch or total')

        # ==========================================
        # For run_vln_pretraining in Oscar
        self.parser.add_argument("--data_dir", default='datasets/coco_caption', type=str, required=False,
                            help="The input data dir with all required files.")
        self.parser.add_argument("--train_yaml", default='/media/diskpart2/oscar_data/r2r_vln/train.yaml', type=str,
                            required=False,
                            help="yaml file for training.")
        self.parser.add_argument("--test_yaml", default='test.yaml', type=str, required=False,
                            help="yaml file for testing.")
        self.parser.add_argument("--val_yaml", default='val.yaml', type=str, required=False,
                            help="yaml file used for validation during training.")
        self.parser.add_argument("--model_name_or_path", default=None, type=str, required=False,
                            help="Path to pre-trained model or model type.")
        self.parser.add_argument("--output_dir", default='output_r2r/', type=str, required=False,
                            help="The output directory to save checkpoint and test results.")
        self.parser.add_argument("--loss_type", default='sfmx', type=str,
                            help="Loss function types: support kl, x2, sfmx")
        self.parser.add_argument("--config_name", default="", type=str,
                            help="Pretrained config name or path if not the same as model_name.")
        self.parser.add_argument("--tokenizer_name", default="", type=str,
                            help="Pretrained tokenizer name or path if not the same as model_name.")
        self.parser.add_argument("--max_seq_length", default=180, type=int,
                            help="The maximum total input sequence length after tokenization. "
                                 "Sequences longer than this will be truncated, "
                                 "sequences shorter will be padded.")
        self.parser.add_argument("--max_seq_a_length", default=80, type=int,
                            help="The maximum sequence length for caption.")
        self.parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
        self.parser.add_argument("--do_test", action='store_true', help="Whether to run inference.")
        self.parser.add_argument("--do_eval", action='store_true', help="Whether to run evaluation.")
        self.parser.add_argument("--do_lower_case", action='store_true',
                            help="Set this flag if you are using an uncased model.")
        self.parser.add_argument("--mask_prob", default=0.15, type=float,
                            help="Probability to mask input sentence during training.")
        self.parser.add_argument("--max_masked_tokens", type=int, default=4,
                            help="The max number of masked tokens per sentence.")
        self.parser.add_argument("--add_od_labels", default=False, action='store_true',
                            help="Whether to add object detection labels or not")
        self.parser.add_argument("--drop_out", default=0.1, type=float, help="Drop out in BERT.")
        self.parser.add_argument("--max_img_seq_length", default=100, type=int,
                            help="The maximum total input image sequence length.")
        self.parser.add_argument("--model_img_feature_dim", default=2054, type=int,
                            help="The Image Feature Dimension.")
        self.parser.add_argument("--img_feature_type", default='frcnn', type=str,
                            help="Image feature type.")
        # self.parser.add_argument("--per_gpu_train_batch_size", default=64, type=int,
        #                     help="Batch size per GPU/CPU for training.")
        self.parser.add_argument("--per_gpu_eval_batch_size", default=64, type=int,
                            help="Batch size per GPU/CPU for evaluation.")
        self.parser.add_argument("--output_mode", default='classification', type=str,
                            help="output mode, support classification or regression.")
        self.parser.add_argument("--num_labels", default=2, type=int,
                            help="num_labels is 2 for classification and 1 for regression.")
        # self.parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
        #                     help="Number of updates steps to accumulate before backward.")
        self.parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial lr.")
        self.parser.add_argument("--weight_decay", default=0.05, type=float, help="Weight deay.")
        self.parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam.")
        self.parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
        self.parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup.")
        self.parser.add_argument("--scheduler", default='linear', type=str, help="constant or linear or")
        # self.parser.add_argument("--num_workers", default=4, type=int, help="Workers in dataloader.")
        # self.parser.add_argument("--num_train_epochs", default=40, type=int,
        #                     help="Total number of training epochs to perform.")
        self.parser.add_argument('--logging_steps', type=int, default=20, help="Log every X steps.")
        self.parser.add_argument('--save_steps', type=int, default=-1,
                            help="Save checkpoint every X steps. Will also perform evaluatin.")
        self.parser.add_argument("--evaluate_during_training", action='store_true',
                            help="Run evaluation during training at each save_steps.")
        self.parser.add_argument("--no_cuda", action='store_true', help="Avoid using CUDA.")
        self.parser.add_argument('--seed', type=int, default=88, help="random seed for initialization.")
        self.parser.add_argument('--scst', action='store_true', help='Self-critical sequence training')
        # for generation
        self.parser.add_argument("--eval_model_dir", type=str, default='',
                            help="Model directory for evaluation.")
        self.parser.add_argument('--max_gen_length', type=int, default=20,
                            help="max length of generated sentences")
        self.parser.add_argument('--output_hidden_states', action='store_true',
                            help="Turn on for fast decoding")
        self.parser.add_argument('--num_return_sequences', type=int, default=1,
                            help="repeating times per image")
        self.parser.add_argument('--num_beams', type=int, default=5, help="beam search width")
        self.parser.add_argument('--num_keep_best', type=int, default=1,
                            help="number of hypotheses to keep in beam search")
        self.parser.add_argument('--temperature', type=float, default=1,
                            help="temperature in softmax for sampling")
        self.parser.add_argument('--top_k', type=int, default=0,
                            help="filter distribution for sampling")
        self.parser.add_argument('--top_p', type=float, default=1,
                            help="filter distribution for sampling")
        self.parser.add_argument('--repetition_penalty', type=int, default=1,
                            help="repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)")
        self.parser.add_argument('--length_penalty', type=int, default=1,
                            help="beam search length penalty")
        # for Constrained Beam Search
        self.parser.add_argument('--use_cbs', action='store_true',
                            help='Use constrained beam search for decoding')
        self.parser.add_argument('--min_constraints_to_satisfy', type=int, default=2,
                            help="minimum number of constraints to satisfy")
        # ==========================================

        self.args = self.parser.parse_args()

        if self.args.optim == 'rms':
            print("param_r2r_oscar Optimizer: Using RMSProp")
            # self.args.optimizer = torch.optim.RMSprop
        elif self.args.optim == 'adam':
            # print("param_r2r_oscar Optimizer: Using Adam")
            self.args.optimizer = torch.optim.Adam
        elif self.args.optim == 'sgd':
            # print("param_r2r_oscar Optimizer: sgd")
            self.args.optimizer = torch.optim.SGD
        else:
            assert False

param = Param()
r2r_envdrop_args = param.args
r2r_envdrop_args.TRAIN_VOCAB = 'r2r_envdrop/tasks/R2R/data/train_vocab.txt'
r2r_envdrop_args.TRAINVAL_VOCAB = 'r2r_envdrop/tasks/R2R/data/trainval_vocab.txt'

r2r_envdrop_args.IMAGENET_FEATURES = 'r2r_envdrop/img_features/ResNet-152-imagenet.tsv'
r2r_envdrop_args.FASTERRCNN_FEATURES = '/media/diskpart1/VLN_Data/img_features/faster_rcnn_end2end_total.tsv'

r2r_envdrop_args.features = r2r_envdrop_args.IMAGENET_FEATURES

r2r_envdrop_args.log_dir = 'snap/%s' % r2r_envdrop_args.name

r2r_envdrop_args.IMG_WIDTH = 640
r2r_envdrop_args.IMG_HEIGHT = 480

r2r_envdrop_args.max_objects_panorama = 100

if not os.path.exists(r2r_envdrop_args.log_dir):
    os.makedirs(r2r_envdrop_args.log_dir)
DEBUG_FILE = open(os.path.join('snap', r2r_envdrop_args.name, "debug.log"), 'w')

