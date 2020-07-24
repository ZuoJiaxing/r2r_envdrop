''' Batched Room-to-Room navigation environment '''

import sys
import os
sys.path.append('build')
base_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.dirname(base_path)
# sys.path.append(base_path)
sys.path.append(project_path + '/build')
import MatterSim
import csv
import numpy as np
import math
import base64
import json
import os
import random
import networkx as nx
from .param import r2r_envdrop_args as args
from .utils import load_datasets, load_nav_graphs, Tokenizer
from . import utils
# from oscar.run_vln_pretraining import imgfeat_r2r

csv.field_size_limit(sys.maxsize)


class EnvBatch():
    ''' A simple wrapper for a batch of MatterSim environments, 
        using discretized viewpoints and pretrained features '''

    def __init__(self, feature_store=None, feature_navp_store=None,batch_size=100):
        """
        1. Load pretrained image feature
        2. Init the Simulator.
        :param feature_store: The name of file stored the feature.
        :param batch_size:  Used to create the simulator list.
        """
        if feature_store:
            self.features = feature_store
            self.image_w = 640
            self.image_h = 480
            self.vfov = 60
            self.feature_size =  self.features.get_feature_size()
            print('The feature size is %d' % self.feature_size)

            self.features_navp = feature_navp_store

        self.featurized_scans = feature_store.get_feat_scans()
        self.sims = []
        for i in range(batch_size):
            sim = MatterSim.Simulator()
            sim.setRenderingEnabled(False)
            sim.setDiscretizedViewingAngles(True)   # Set increment/decrement to 30 degree. (otherwise by radians)
            sim.setCameraResolution(self.image_w, self.image_h)
            sim.setCameraVFOV(math.radians(self.vfov))
            sim.init()
            self.sims.append(sim)

    def _make_id(self, scanId, viewpointId):
        return scanId + '_' + viewpointId   

    def newEpisodes(self, scanIds, viewpointIds, headings):
        for i, (scanId, viewpointId, heading) in enumerate(zip(scanIds, viewpointIds, headings)):
            # print("New episode %d" % i)
            # sys.stdout.flush()
            self.sims[i].newEpisode(scanId, viewpointId, heading, 0)
  
    def getStates(self):
        """
        Get list of states augmented with precomputed image features. rgb field will be empty.
        Agent's current view [0-35] (set only when viewing angles are discretized)
            [0-11] looking down, [12-23] looking at horizon, [24-35] looking up
        :return: [ ((30, 2048), sim_state) ] * batch_size
        """
        feature_label_states = []
        for i, sim in enumerate(self.sims):
            state = sim.getState()

            long_id = self._make_id(state.scanId, state.location.viewpointId)
            if self.features:
                feature,label = self.features.get_image_features(long_id)     # Get feature for
                feature_label_states.append((feature, label, state))
            else:
                feature_label_states.append((None, None, state))
        return feature_label_states

    def getNavpFeat(self):
        """
        Get dict of navigable region features in subimages
        """
        navp_feature_states = []
        for i, sim in enumerate(self.sims):
            state = sim.getState()

            long_id = self._make_id(state.scanId, state.location.viewpointId)
            if self.features:
                feature_dict= self.features_navp.get_navp_features(long_id)     # Get feature for
                navp_feature_states.append((feature_dict, state))
            else:
                navp_feature_states.append((None, state))
        return navp_feature_states

    def makeActions(self, actions):
        ''' Take an action using the full state dependent action interface (with batched input). 
            Every action element should be an (index, heading, elevation) tuple. '''
        for i, (index, heading, elevation) in enumerate(actions):
            self.sims[i].makeAction(index, heading, elevation)

class R2RBatch():
    ''' Implements the Room to Room navigation task, using discretized viewpoints and pretrained features '''

    def __init__(self, feature_store, feature_navp_store, batch_size=100, seed=10, splits=['train'], tokenizer=None,
                 name=None):
        self.env = EnvBatch(feature_store=feature_store, feature_navp_store=feature_navp_store, batch_size=batch_size)
        if feature_store:
            self.feature_size = self.env.feature_size
        self.data = []
        if tokenizer:
            self.tok = tokenizer
        scans = []
        for split in splits:
            for item in load_datasets([split]):
                # Split multiple instructions into separate entries
                for j,instr in enumerate(item['instructions']):
                    if item['scan'] not in self.env.featurized_scans:   # For fast training
                        continue
                    new_item = dict(item)
                    new_item['instr_id'] = '%s_%d' % (item['path_id'], j)
                    new_item['instructions'] = instr
                    self.data.append(new_item)
                    scans.append(item['scan'])
        if name is None:
            self.name = splits[0] if len(splits) > 0 else "FAKE"
        else:
            self.name = name

        self.scans = set(scans)
        self.splits = splits
        self.seed = seed
        random.seed(self.seed)
        random.shuffle(self.data)

        self.ix = 0
        self.batch_size = batch_size
        self._load_nav_graphs()

        self.angle_feature = utils.get_all_point_angle_feature()
        self.sim = utils.new_simulator()
        self.buffered_state_dict = {}

        # It means that the fake data is equals to data in the supervised setup
        self.fake_data = self.data
        print('R2RBatch loaded with %d instructions, using splits: %s' % (len(self.data), ",".join(splits)))

    def get_ob_feture_size(self):
        return self.feature_size -1 + args.angle_feat_size

    def size(self):
        return len(self.data)

    def __len__(self):
        return len(self.data)

    def _load_nav_graphs(self):
        """
        load graph from self.scan,
        Store the graph {scan_id: graph} in self.graphs
        Store the shortest path {scan_id: {view_id_x: {view_id_y: [path]} } } in self.paths
        Store the distances in self.distances. (Structure see above)
        Load connectivity graph for each scan, useful for reasoning about shortest paths
        :return: None
        """
        print('Loading navigation graphs for %d scans' % len(self.scans))
        self.graphs = load_nav_graphs(self.scans)
        self.paths = {}
        for scan, G in self.graphs.items(): # compute all shortest paths
            self.paths[scan] = dict(nx.all_pairs_dijkstra_path(G))
        self.distances = {}
        for scan, G in self.graphs.items(): # compute all shortest paths
            self.distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

    def _next_minibatch(self, tile_one=False, batch_size=None, **kwargs):
        """
        Store the minibach in 'self.batch'
        :param tile_one: Tile the one into batch_size
        :return: None
        """
        # print("for a test, self.ix: ", self.ix)
        if batch_size is None:
            batch_size = self.batch_size
        if tile_one:
            batch = [self.data[self.ix]] * batch_size
            self.ix += 1
            if self.ix >= len(self.data):
                random.shuffle(self.data)
                self.ix -= len(self.data)
        else:
            batch = self.data[self.ix: self.ix+batch_size]
            if len(batch) < batch_size:
                random.shuffle(self.data)
                self.ix = batch_size - len(batch)
                batch += self.data[:self.ix]
            else:
                self.ix += batch_size
        self.batch = batch

    def reset_epoch(self, shuffle=False):
        ''' Reset the data index to beginning of epoch. Primarily for testing. 
            You must still call reset() for a new episode. '''
        if shuffle:
            random.shuffle(self.data)
        self.ix = 0

    def _shortest_path_action(self, state, goalViewpointId):
        ''' Determine next action on the shortest path to goal, for supervised training. '''
        if state.location.viewpointId == goalViewpointId:
            return goalViewpointId      # Just stop here
        path = self.paths[state.scanId][state.location.viewpointId][goalViewpointId]
        nextViewpointId = path[1]
        return nextViewpointId

    # The input feature is np array with shpae N*2055, and the last dimension is subimg_idx
    # Note that, we return the feature with dimension 2044, and cut off the original subimg_idx
    def _get_feat_insubimg(self,feature,subimg_idx):
        out_feat = []
        for feat_block in feature:
            # The last dimension of the feat_block is the subimg idx
            if feat_block[-1] == subimg_idx:
                out_feat.append(feat_block[0:-1])
        return np.array(out_feat)

    # The input feature is np array with shpae N*2055, and the last dimension is subimg_idx
    # The input label is the list of objects labels with length N
    # Note that, we return the feature with dimension 2044, and cut off the original subimg_idx
    # And also, return the corresponding object lable
    def _get_feat_label_insubimg(self,feature, label, subimg_idx):
        assert(len(feature) == len(label))
        out_feat = []
        out_label = []
        for feat_block,label_block in zip(feature,label):
            # The last dimension of the feat_block is the subimg idx
            if feat_block[-1] == subimg_idx:
                out_feat.append(feat_block[0:-1])
                out_label.append(label_block)
        # print("subimg_idx", subimg_idx)
        assert(out_feat)
        return np.array(out_feat), out_label

    # The input feature is np array with shpae N*2055, and the last dimension is subimg_idx
    # Cut off the last subimg_idx dimension and return np arry with N*2054
    def _shorten_feat(self, feature):
        return feature[:,:-1]
    
    # The input angle_feature is of dimension 36*args.angle_feat_size
    def _shorten_feat_add_anglefeat(self, feature, angle_feature):
        outfeature_subimg_idx = feature[:,-1]
        outfeature = np.empty((len(feature), feature.shape[-1] + angle_feature.shape[-1] - 1), np.float32)
        for idx, feat in enumerate(feature):
            outfeature[idx,:] = np.concatenate((feat[:-1], angle_feature[int(feat[-1])]), -1)
        return outfeature,outfeature_subimg_idx

    # Rocky: Check out the candidates action based on current status
    # The input feature is np array with shpae N*2055
    # The input navp_feature_dict is dict with {long_imgid+'nav_idx': feature with 2055}
    def make_candidate(self, feature, navp_feature_dict, label, scanId, viewpointId, viewId):

        # print("===========navp_feature_dict"*3)
        # print(navp_feature_dict)

        def _loc_distance(loc):
            return np.sqrt(loc.rel_heading ** 2 + loc.rel_elevation ** 2)
        base_heading = (viewId % 12) * math.radians(30)
        adj_dict = {}
        long_id = "%s_%s" % (scanId, viewpointId)
        if long_id not in self.buffered_state_dict:
            """
            Agent's current view [0-35] (set only when viewing angles are discretized)
            [0-11] looking down, [12-23] looking at horizon, [24-35] looking up
            
            Rocky: from 0 to 35, look up the accessiable candidates.
            """
            for ix in range(36):
                if ix == 0:
                    # Rocky:  newEpisode(scanId, viewpointId, heading, elevation);
                    self.sim.newEpisode(scanId, viewpointId, 0, math.radians(-30))
                elif ix % 12 == 0:
                    #  Rocky: makeAction(index, heading, elevation);
                    self.sim.makeAction(0, 1.0, 1.0)
                else:
                    self.sim.makeAction(0, 1.0, 0)

                state = self.sim.getState()
                assert state.viewIndex == ix

                # Heading and elevation for the viewpoint center
                heading = state.heading - base_heading
                elevation = state.elevation

                # Get the features of a subimage from a np.array
                # visual_feat,visual_label = self._get_feat_label_insubimg(feature, label, ix)

                # get adjacent locations
                for j, loc in enumerate(state.navigableLocations[1:]):
                    # if a loc is visible from multiple view, use the closest
                    # view (in angular distance) as its representation
                    distance = _loc_distance(loc)

                    # Heading and elevation for for the loc
                    loc_heading = heading + loc.rel_heading
                    loc_elevation = elevation + loc.rel_elevation
                    # The dimension of angle_feat is determined by args.angle_feat_size, the default is 1*4
                    angle_feat = utils.angle_feature(loc_heading, loc_elevation)

                    # Try to pick out the region feature of navigable point
                    visual_feat = np.zeros((1, 2054), dtype=np.float32)
                    navloc_id = long_id + '_' + str(ix) + '_' + str(j)
                    if navloc_id in navp_feature_dict:
                        visual_feat = navp_feature_dict[navloc_id][:-1]
                        visual_feat = np.reshape(visual_feat, (1,len(visual_feat)))
                        # print("Get navp feat id with shape: ", navloc_id, visual_feat.shape)
                    else:
                        # print("Fail to get navp feat id: ", navloc_id)
                        pass

                    if (loc.viewpointId not in adj_dict or
                            distance < adj_dict[loc.viewpointId]['distance']):
                        # print("visual_feat.shape", visual_feat.shape)
                        # print("np.tile(angle_feat,(len(visual_feat),1)).shape", (np.tile(angle_feat,(len(visual_feat),1))).shape)

                        adj_dict[loc.viewpointId] = {
                            'heading': loc_heading,
                            'elevation': loc_elevation,
                            'normalized_heading': state.heading + loc.rel_heading,
                            'scanId':scanId,
                            'viewpointId': loc.viewpointId, # Next viewpoint id
                            'pointId': ix,
                            'distance': distance,
                            'idx': j + 1,
                            'navloc_id': navloc_id,
                            # 'feature': np.concatenate((visual_feat, np.tile(angle_feat,(len(visual_feat),1))), -1), # With dimension X*(2054 + args.angle_feat_size)
                            'feature': np.concatenate((visual_feat, np.tile(angle_feat,(len(visual_feat),1))), -1), # With dimension 1*(2054 + args.angle_feat_size)
                            'label': 'point'
                        }
            candidate = list(adj_dict.values())
            # print("For a test, the navp_feature_dict size: ", len(navp_feature_dict))
            # print("For a test, the candidate size: ", len(candidate))
            # for cd in candidate:
            #     print("norm: ", np.linalg.norm(cd['feature'][:,0:2048]))

            self.buffered_state_dict[long_id] = [
                {key: c[key]
                 for key in
                    ['normalized_heading', 'elevation', 'scanId', 'viewpointId',
                     'pointId', 'idx', 'navloc_id']}
                for c in candidate
            ]
            return candidate
        else:
            candidate = self.buffered_state_dict[long_id]
            candidate_new = []
            # todo: update the label as the above
            for c in candidate:
                c_new = c.copy()
                # ix = c_new['pointId']
                normalized_heading = c_new['normalized_heading']

                navloc_id = c_new['navloc_id']
                # Try to pick out the region feature of navigable point
                visual_feat = np.zeros((1, 2054), dtype=np.float32)
                if navloc_id in navp_feature_dict:
                    visual_feat = navp_feature_dict[navloc_id][:-1]
                    visual_feat = np.reshape(visual_feat, (1, len(visual_feat)))
                else:
                    # print("Fail to get navp feat id: ", navloc_id)
                    pass
                # visual_feat = self._get_feat_insubimg(feature,ix)
                loc_heading = normalized_heading - base_heading
                c_new['heading'] = loc_heading
                angle_feat = utils.angle_feature(c_new['heading'], c_new['elevation'])
                c_new['feature'] = np.concatenate((visual_feat, np.tile(angle_feat,(len(visual_feat),1))), -1)
                c_new['label'] = 'point'
                c_new.pop('normalized_heading')
                candidate_new.append(c_new)
            return candidate_new

    def _get_obs(self):
        obs = []
        batch_navp_feats = self.env.getNavpFeat() # list of (feature_dict, state)
        for i, (feature, label, state) in enumerate(self.env.getStates()):
            assert(len(feature) == len(label))
            item = self.batch[i]
            base_view_id = state.viewIndex

            navp_feature_dict = batch_navp_feats[i][0]
            # print("for a test, len(navp_feature): ", len(navp_feature_dict))

            # Full features
            candidate = self.make_candidate(feature, navp_feature_dict, label, state.scanId, state.location.viewpointId, state.viewIndex)

            # # Check whether the 'pointId' in candidate is unique
            # tmp_pointId_list = [c['pointId'] for c in candidate]
            # # ERROR, because there might be multiple navigable points from a subimg view
            # assert len(tmp_pointId_list) == len(set(tmp_pointId_list))

            # (visual_feature, angel_feature) for views
            # The shape of self.angle_feature[base_view_id] is 36*args.angle_feat_size
            feature, feature_subimg_idx = self._shorten_feat_add_anglefeat(feature, self.angle_feature[base_view_id])
            feature_subimg_idx=feature_subimg_idx.astype(int)
            obs.append({
                'instr_id' : item['instr_id'],
                'scan' : state.scanId,
                'viewpoint' : state.location.viewpointId,
                'viewIndex' : state.viewIndex,
                'heading' : state.heading,
                'elevation' : state.elevation,
                'feature' : feature,
                'feature_subimg_idx' : feature_subimg_idx,
                'label' : label,
                'candidate': candidate,
                'navigableLocations' : state.navigableLocations,
                'instructions' : item['instructions'],
                'teacher' : self._shortest_path_action(state, item['path'][-1]),
                'path_id' : item['path_id']
            })

            # A2C reward. The negative distance between the state and the final state
            obs[-1]['distance'] = self.distances[state.scanId][state.location.viewpointId][item['path'][-1]]
        return obs

    def reset(self, batch=None, inject=False, **kwargs):
        ''' Load a new minibatch / episodes. '''
        if batch is None:       # Allow the user to explicitly define the batch
            self._next_minibatch(**kwargs)
        else:
            if inject:          # Inject the batch into the next minibatch
                self._next_minibatch(**kwargs)
                self.batch[:len(batch)] = batch
            else:               # Else set the batch to the current batch
                self.batch = batch
        scanIds = [item['scan'] for item in self.batch]
        viewpointIds = [item['path'][0] for item in self.batch]
        headings = [item['heading'] for item in self.batch]
        self.env.newEpisodes(scanIds, viewpointIds, headings)
        return self._get_obs()

    def step(self, actions):
        ''' Take action (same interface as makeActions) '''
        self.env.makeActions(actions)
        return self._get_obs()

    def get_statistics(self):
        stats = {}
        length = 0
        path = 0
        for datum in self.data:
            # length += len(self.tok.split_sentence(datum['instructions']))
            length += len(datum['instructions'].strip().split())
            path += self.distances[datum['scan']][datum['path'][0]][datum['path'][-1]]
        stats['length'] = length / len(self.data)
        stats['path'] = path / len(self.data)
        return stats
