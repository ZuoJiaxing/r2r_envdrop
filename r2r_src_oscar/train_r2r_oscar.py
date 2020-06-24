
import torch

import os
import time
import json
import numpy as np
from collections import defaultdict


import sys
base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_path)

from utils import read_vocab,write_vocab,build_vocab,Tokenizer,padding_idx,timeSince, read_img_features
from env_oscar import R2RBatch
from agent import Seq2SeqAgent
from eval import Evaluation
from param import r2r_envdrop_args as args

from oscar.run_vln_pretraining import imgfeat_r2r

def train(train_env, tok, n_iters, log_every=100, val_envs={}, aug_env=None):
    writer = SummaryWriter(logdir=args.log_dir)
    listner = Seq2SeqAgent(train_env, "", tok, args.maxAction)

    speaker = None
    if args.self_train:
        speaker = Speaker(train_env, listner, tok)
        if args.speaker is not None:
            print("Load the speaker from %s." % args.speaker)
            speaker.load(args.speaker)

    start_iter = 0
    if args.load is not None:
        print("LOAD THE listener from %s" % args.load)
        start_iter = listner.load(os.path.join(args.load))

    start = time.time()

    best_val = {'val_seen': {"accu": 0., "state":"", 'update':False},
                'val_unseen': {"accu": 0., "state":"", 'update':False}}
    if args.fast_train:
        log_every = 40
    for idx in range(start_iter, start_iter+n_iters, log_every):
        listner.logs = defaultdict(list)
        interval = min(log_every, n_iters-idx)
        iter = idx + interval

        # Train for log_every interval
        if aug_env is None:     # The default training process
            listner.env = train_env
            listner.train(interval, feedback=args.feedback)   # Train interval iters
        else:
            if args.accumulate_grad:
                for _ in range(interval // 2):
                    listner.zero_grad()
                    listner.env = train_env

                    # Train with GT data
                    args.ml_weight = 0.2
                    listner.accumulate_gradient(args.feedback)
                    listner.env = aug_env

                    # Train with Back Translation
                    args.ml_weight = 0.6        # Sem-Configuration
                    listner.accumulate_gradient(args.feedback, speaker=speaker)
                    listner.optim_step()
            else:
                for _ in range(interval // 2):
                    # Train with GT data
                    listner.env = train_env
                    args.ml_weight = 0.2
                    listner.train(1, feedback=args.feedback)

                    # Train with Back Translation
                    listner.env = aug_env
                    args.ml_weight = 0.6
                    listner.train(1, feedback=args.feedback, speaker=speaker)

        # Log the training stats to tensorboard
        total = max(sum(listner.logs['total']), 1)
        length = max(len(listner.logs['critic_loss']), 1)
        critic_loss = sum(listner.logs['critic_loss']) / total #/ length / args.batchSize
        entropy = sum(listner.logs['entropy']) / total #/ length / args.batchSize
        predict_loss = sum(listner.logs['us_loss']) / max(len(listner.logs['us_loss']), 1)
        writer.add_scalar("loss/critic", critic_loss, idx)
        writer.add_scalar("policy_entropy", entropy, idx)
        writer.add_scalar("loss/unsupervised", predict_loss, idx)
        writer.add_scalar("total_actions", total, idx)
        writer.add_scalar("max_length", length, idx)
        print("total_actions", total)
        print("max_length", length)

        # Run validation
        loss_str = ""
        for env_name, (env, evaluator) in val_envs.items():
            listner.env = env

            # Get validation loss under the same conditions as training
            iters = None if args.fast_train or env_name != 'train' else 20     # 20 * 64 = 1280

            # Get validation distance from goal under test evaluation conditions
            listner.test(use_dropout=False, feedback='argmax', iters=iters)
            result = listner.get_results()
            score_summary, _ = evaluator.score(result)
            loss_str += ", %s " % env_name
            for metric,val in score_summary.items():
                if metric in ['success_rate']:
                    writer.add_scalar("accuracy/%s" % env_name, val, idx)
                    if env_name in best_val:
                        if val > best_val[env_name]['accu']:
                            best_val[env_name]['accu'] = val
                            best_val[env_name]['update'] = True
                loss_str += ', %s: %.3f' % (metric, val)

        for env_name in best_val:
            if best_val[env_name]['update']:
                best_val[env_name]['state'] = 'Iter %d %s' % (iter, loss_str)
                best_val[env_name]['update'] = False
                listner.save(idx, os.path.join("snap", args.name, "state_dict", "best_%s" % (env_name)))

        print(('%s (%d %d%%) %s' % (timeSince(start, float(iter)/n_iters),
                                             iter, float(iter)/n_iters*100, loss_str)))

        if iter % 1000 == 0:
            print("BEST RESULT TILL NOW")
            for env_name in best_val:
                print(env_name, best_val[env_name]['state'])

        if iter % 50000 == 0:
            listner.save(idx, os.path.join("snap", args.name, "state_dict", "Iter_%06d" % (iter)))

    listner.save(idx, os.path.join("snap", args.name, "state_dict", "LAST_iter%d" % (idx)))


def setup():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    # Check for vocabs
    if not os.path.exists(args.TRAIN_VOCAB):
        write_vocab(build_vocab(splits=['train']), args.TRAIN_VOCAB)
    if not os.path.exists(args.TRAINVAL_VOCAB):
        write_vocab(build_vocab(splits=['train','val_seen','val_unseen']), args.TRAINVAL_VOCAB)

def prepare_r2r_data():
    ''' Prepare data from the training set, valseen and val_unseen splits. '''
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    feature_data = imgfeat_r2r('/media/diskpart2/oscar_data/r2r_vln/train.yaml')
    featurized_scans = feature_data.get_feat_scans()
    train_env = R2RBatch(feature_data, batch_size=args.batchSize, splits=['train'])
    listner = Seq2SeqAgent(train_env, "", tok, args.maxAction)
    listner.env = train_env
    listner.train(interval, feedback=args.feedback)  # Train interval iters







def train_val():
    ''' Train on the training set, and validate on seen and unseen splits. '''
    # args.fast_train = True
    setup()


    # Create a batch training environment that will also preprocess text
    vocab = read_vocab(args.TRAIN_VOCAB) # A word list
    tok = Tokenizer(vocab=vocab, encoding_length=args.maxInput)

    feat_dict = read_img_features(args.features) # A dict {long_id, feat}

    featurized_scans = set([key.split("_")[0] for key in list(feat_dict.keys())])

    train_env = R2RBatch(feat_dict, batch_size=args.batchSize, splits=['train'], tokenizer=tok)
    from collections import OrderedDict

    val_env_names = ['val_unseen', 'val_seen']
    if args.submit:
        val_env_names.append('test')
    else:
        pass
        #val_env_names.append('train')

    if not args.beam:
        val_env_names.append("train")

    val_envs = OrderedDict(
        ((split,
          (R2RBatch(feat_dict, batch_size=args.batchSize, splits=[split], tokenizer=tok),
           Evaluation([split], featurized_scans, tok))
          )
         for split in val_env_names
         )
    )

    if args.train == 'listener':
        train(train_env, tok, args.iters, val_envs=val_envs)
    else:
        assert False

def train_val_augment():
    """
    Train the listener with the augmented data
    """
    setup()

    # Create a batch training environment that will also preprocess text
    vocab = read_vocab(args.TRAIN_VOCAB)
    tok = Tokenizer(vocab=vocab, encoding_length=args.maxInput)

    # Load the env img features
    feat_dict = read_img_features(args.features)
    featurized_scans = set([key.split("_")[0] for key in list(feat_dict.keys())])

    # Load the augmentation data
    aug_path = args.aug

    # Create the training environment
    train_env = R2RBatch(feat_dict, batch_size=args.batchSize,
                         splits=['train'], tokenizer=tok)
    aug_env   = R2RBatch(feat_dict, batch_size=args.batchSize,
                         splits=[aug_path], tokenizer=tok, name='aug')

    # Printing out the statistics of the dataset
    stats = train_env.get_statistics()
    print("The training data_size is : %d" % train_env.size())
    print("The average instruction length of the dataset is %0.4f." % (stats['length']))
    print("The average action length of the dataset is %0.4f." % (stats['path']))
    stats = aug_env.get_statistics()
    print("The augmentation data size is %d" % aug_env.size())
    print("The average instruction length of the dataset is %0.4f." % (stats['length']))
    print("The average action length of the dataset is %0.4f." % (stats['path']))

    # Setup the validation data
    val_envs = {split: (R2RBatch(feat_dict, batch_size=args.batchSize, splits=[split],
                                 tokenizer=tok), Evaluation([split], featurized_scans, tok))
                for split in ['train', 'val_seen', 'val_unseen']}

    # Start training
    train(train_env, tok, args.iters, val_envs=val_envs, aug_env=aug_env)

#
# if __name__ == "__main__":
#     if args.train in ['listener']:
#         train_val()
#     elif args.train == 'auglistener':
#         train_val_augment()
#     else:
#         assert False

