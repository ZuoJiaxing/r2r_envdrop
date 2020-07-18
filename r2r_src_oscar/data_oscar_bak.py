# For getting data from the simulator
import json
import os
import sys
import numpy as np
import random
import math
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_path)
from env_oscar import R2RBatch
from utils import padding_idx, add_idx, Tokenizer
import utils
import model
from param import r2r_envdrop_args as args
from collections import defaultdict
from oscar.run_vln_pretraining import imgfeat_r2r


class BaseData(object):
    ''' Base class for an R2R agent to generate and save trajectories. '''

    def __init__(self, env, results_path):
        self.env = env
        self.results_path = results_path
        random.seed(1)
        self.results = {}
        self.losses = []  # For learning agents

    def write_results(self):
        output = [{'instr_id': k, 'trajectory': v} for k, v in self.results.items()]
        with open(self.results_path, 'w') as f:
            json.dump(output, f)

    def get_results(self):
        output = [{'instr_id': k, 'trajectory': v} for k, v in self.results.items()]
        return output

    def rollout(self, **args):
        ''' Return a list of dicts containing instr_id:'xx', path:[(viewpointId, heading_rad, elevation_rad)]  '''
        raise NotImplementedError

    @staticmethod
    def get_agent(name):
        return globals()[name + "Agent"]

    def test(self, iters=None, **kwargs):
        self.env.reset_epoch(shuffle=(iters is not None))  # If iters is not none, shuffle the env batch
        self.losses = []
        self.results = {}
        # We rely on env showing the entire batch before repeating anything
        looped = False
        self.loss = 0
        if iters is not None:
            # For each time, it will run the first 'iters' iterations. (It was shuffled before)
            for i in range(iters):
                for traj in self.rollout(**kwargs):
                    self.loss = 0
                    self.results[traj['instr_id']] = traj['path']
        else:  # Do a full round
            while True:
                for traj in self.rollout(**kwargs):
                    if traj['instr_id'] in self.results:
                        looped = True
                    else:
                        self.loss = 0
                        self.results[traj['instr_id']] = traj['path']
                if looped:
                    break


class DataManager(BaseData):
    ''' A data manager to organize the r2r data '''
    # For now, the agent can't pick which forward move to make - just the one in the middle
    env_actions = {
      'left': (0,-1, 0), # left
      'right': (0, 1, 0), # right
      'up': (0, 0, 1), # up
      'down': (0, 0,-1), # down
      'forward': (1, 0, 0), # forward
      '<end>': (0, 0, 0), # <end>
      '<start>': (0, 0, 0), # <start>
      '<ignore>': (0, 0, 0)  # <ignore>
    }

    def __init__(self, env, results_path, episode_len=20):
        super(DataManager, self).__init__(env, results_path)
        self.episode_len = episode_len
        self.feature_size = self.env.feature_size

        # Logs
        sys.stdout.flush()
        self.logs = defaultdict(list)

    def _feature_variable(self, obs):
        ''' Extract precomputed features into variable. '''
        features = np.empty((len(obs), args.views, self.feature_size + args.angle_feat_size), dtype=np.float32)
        for i, ob in enumerate(obs):
            features[i, :, :] = ob['feature']   # Image feat
        # return Variable(torch.from_numpy(features), requires_grad=False).cuda()
        return features

    def _candidate_variable(self, obs):
        candidate_leng = [len(ob['candidate']) + 1 for ob in obs]       # +1 is for the end
        candidate_feat = np.zeros((len(obs), max(candidate_leng), self.feature_size + args.angle_feat_size), dtype=np.float32)
        # Note: The candidate_feat at len(ob['candidate']) is the feature for the END
        # which is zero in my implementation
        for i, ob in enumerate(obs):
            for j, c in enumerate(ob['candidate']):
                candidate_feat[i, j, :] = c['feature']                         # Image feat
        # return torch.from_numpy(candidate_feat).cuda(), candidate_leng
        return candidate_feat, candidate_leng

    def get_input_feat(self, obs):
        input_a_t = np.zeros((len(obs), args.angle_feat_size), np.float32)
        for i, ob in enumerate(obs):
            input_a_t[i] = utils.angle_feature(ob['heading'], ob['elevation'])
        # input_a_t = torch.from_numpy(input_a_t).cuda()

        f_t = self._feature_variable(obs)      # Image features from obs
        candidate_feat, candidate_leng = self._candidate_variable(obs)

        return input_a_t, f_t, candidate_feat, candidate_leng

    def get_candidate_leng(self,obs):
        candidate_leng = [len(ob['candidate']) + 1 for ob in obs]
        return candidate_leng

    def _teacher_action(self, obs, ended):
        """
        Extract teacher actions into variable.
        :param obs: The observation.
        :param ended: Whether the action seq is ended
        :return:
        """
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:                                            # Just ignore this index
                a[i] = args.ignoreid
            else:
                for k, candidate in enumerate(ob['candidate']):
                    if candidate['viewpointId'] == ob['teacher']:   # Next view point
                        a[i] = k
                        break
                else:   # Stop here
                    assert ob['teacher'] == ob['viewpoint']         # The teacher action should be "STAY HERE"
                    a[i] = len(ob['candidate'])
        return a

    def make_equiv_action(self, a_t, perm_obs, perm_idx=None, traj=None):
        """
        Interface between Panoramic view and Egocentric view
        It will convert the action panoramic view action a_t to equivalent egocentric view actions for the simulator
        """
        def take_action(i, idx, name):
            if type(name) is int:       # Go to the next view
                self.env.env.sims[idx].makeAction(name, 0, 0)
            else:                       # Adjust
                self.env.env.sims[idx].makeAction(*self.env_actions[name])
            state = self.env.env.sims[idx].getState()
            if traj is not None:
                traj[i]['path'].append((state.location.viewpointId, state.heading, state.elevation))
        if perm_idx is None:
            perm_idx = range(len(perm_obs))
        for i, idx in enumerate(perm_idx):
            action = a_t[i]
            if action != -1:            # -1 is the <stop> action
                select_candidate = perm_obs[i]['candidate'][action]
                src_point = perm_obs[i]['viewIndex']
                trg_point = select_candidate['pointId']
                src_level = (src_point ) // 12   # The point idx started from 0
                trg_level = (trg_point ) // 12
                while src_level < trg_level:    # Tune up
                    take_action(i, idx, 'up')
                    src_level += 1
                while src_level > trg_level:    # Tune down
                    take_action(i, idx, 'down')
                    src_level -= 1
                while self.env.env.sims[idx].getState().viewIndex != trg_point:    # Turn right until the target
                    take_action(i, idx, 'right')
                assert select_candidate['viewpointId'] == \
                       self.env.env.sims[idx].getState().navigableLocations[select_candidate['idx']].viewpointId
                take_action(i, idx, select_candidate['idx'])

    def rollout(self, reset=True, feedback = 'teacher'):
        """
        :return:
        """
        self.feedback = feedback
        if reset:
            # Reset env
            obs = np.array(self.env.reset())
        else:
            obs = np.array(self.env._get_obs())

        batch_size = len(obs)
        # Reorder the language input for the encoder (do not ruin the original code)
        perm_obs = obs
        perm_idx = range(len(perm_obs))

        # Record starting point
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])]
        } for ob in perm_obs]

        # For test result submission
        # TODO:
        visited = [set() for _ in perm_obs]

        # Initialization the tracking state
        ended = np.array([False] * batch_size)  # Indices match permuation of the model, not env

        for t in range(self.episode_len):

            # input_a_t, f_t, candidate_feat, candidate_leng = self.get_input_feat(perm_obs)
            candidate_leng = self.get_candidate_leng(perm_obs)

            # Supervised training
            target = self._teacher_action(perm_obs, ended)
            # ml_loss += self.criterion(logit, target)

            # Determine next model inputs
            if self.feedback == 'teacher':
                a_t = target                # teacher forcing
            # elif self.feedback == 'argmax':
            #     _, a_t = logit.max(1)        # student forcing - argmax
            #     a_t = a_t.detach()
            #     log_probs = F.log_softmax(logit, 1)                              # Calculate the log_prob here
            #     policy_log_probs.append(log_probs.gather(1, a_t.unsqueeze(1)))   # Gather the log_prob for each batch
            # elif self.feedback == 'sample':
            #     probs = F.softmax(logit, 1)    # sampling an action from model
            #     c = torch.distributions.Categorical(probs)
            #     self.logs['entropy'].append(c.entropy().sum().item())      # For log
            #     entropys.append(c.entropy())                                # For optimization
            #     a_t = c.sample().detach()
            #     policy_log_probs.append(c.log_prob(a_t))
            else:
                print(self.feedback)
                sys.exit('Invalid feedback option')

            # Prepare environment action
            # NOTE: Env action is in the perm_obs space
            # cpu_a_t = a_t.cpu().numpy()
            cpu_a_t = a_t
            for i, next_id in enumerate(cpu_a_t):
                if next_id == (candidate_leng[i]-1) or next_id == args.ignoreid:    # The last action is <end>
                    cpu_a_t[i] = -1             # Change the <end> and ignore action to -1

            # Make action and get the new state

            self.make_equiv_action(cpu_a_t, perm_obs, perm_idx, traj)
            obs = np.array(self.env._get_obs())
            perm_obs = obs[perm_idx]                    # Perm the obs for the resu

            # Update the finished actions
            # -1 means ended or ignored (already ended)
            ended[:] = np.logical_or(ended, (cpu_a_t == -1))

            # Early exit if all ended
            if ended.all():
                break

        return traj

if __name__ == "__main__":
    ''' Prepare data from the training set, valseen and val_unseen splits. '''
    ''' '''
    torch.manual_seed(1)
    # torch.cuda.manual_seed(1)
    feature_data_store = imgfeat_r2r('/media/diskpart2/oscar_data/r2r_vln/train.yaml')
    featurized_scans = feature_data_store.get_feat_scans()
    train_env = R2RBatch(feature_data_store, batch_size=args.batchSize, splits=['train'])
    data_manager = DataManager(train_env, "abc_tra.txt", args.maxAction)
    data_manager.rollout()
    data_manager.write_results()



