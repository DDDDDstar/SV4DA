import torch
import copy
import os

from .data_preparation import data_prepare
from .nets import RegressionModel, NN
from .utils import DNNTrain, DNNTest


class DV():
    def __init__(self, dataset, manual_seed, GA, TSS):
        self.GA = GA
        self.TSS = TSS
        self.dataset_info = {
            'iris': (3, 4, 30, 16, 0.01),
            'wine': (3, 13, 100, 16, 0.001)
        }
        self.num_classes, self.num_feature, self.ep, self.bs, self.lr  \
            = self.dataset_info[dataset]

        self.model = None
        self.dataset = dataset

        trn_path = 'data/%s0/train0.pt' % (dataset)
        if not os.path.exists(trn_path):
            data_prepare(manual_seed=manual_seed,
                         dataset=dataset,
                         num_classes=self.num_classes)

        # player setting
        self.trn_data = torch.load(trn_path)
        self.players = self.trn_data

        self.Tst = torch.load('data/%s0/test.pt' % (dataset))

    def utility_computation(self, player_list):
        # server for MIA (no use in the other cases)
        player_list = [self.trn_data.idxs[pidx]
                       for pidx in player_list]

        utility = 0.0
        # model initialize and training
        if not self.GA or type(self.model) == type(None) or\
                len(player_list) <= 0:
            if self.GA:
                print('model initialize...')
            if self.dataset == 'iris':
                self.model = RegressionModel(
                    num_feature=self.num_feature,
                    num_classes=self.num_classes)
            elif self.dataset == 'wine':
                self.model = NN(num_feature=self.num_feature,
                                num_classes=self.num_classes)

        if len(player_list) <= 0:
            utility = DNNTest(self.model, self.Tst,
                              metric='tst_accuracy')
            return utility

        loss_func = torch.nn.CrossEntropyLoss()
        if self.GA:
            epoch = 1
            batch_size = 1
            player_list = player_list[-1:]
        else:
            epoch = self.ep
            batch_size = self.bs

        trn_data = copy.deepcopy(self.trn_data)
        trn_data.idxs = player_list  # iterate samples in sub-coalition
        self.model = DNNTrain(self.model, trn_data, self.lr,
                              epoch, batch_size, loss_func)
        del trn_data
        torch.cuda.empty_cache()

        # model testing (maybe expedited by some ML speedup functions)
        utility = DNNTest(self.model, self.Tst,
                          metric='tst_accuracy')

        return utility
