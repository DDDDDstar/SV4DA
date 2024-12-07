import numpy as np
import queue
import os
import copy
import threading
import torch

from functools import reduce

from .data_preparation import data_prepare
from .nets import NN, RegressionModel
from .utils import DNNTrain, DNNTest, find_free_device


class RI():
    def __init__(self, dataset, manual_seed, GA, TSS, parallel_threads_num):
        self.dataset = dataset
        self.manual_seed = manual_seed
        self.GA = GA
        self.TSS = TSS
        self.parallel_threads_num = parallel_threads_num
        self.dataset_info = {
            'iris': (3, 4, 0.01, 30, 16),
            'wine': (3, 13, 0.001, 100, 16)
        }
        self.num_classes, self.num_feature, self.lr, self.ep, self.bs = self.dataset_info[
            dataset]

        self.device = find_free_device()
        self.model = None

        print("Data preprocessing...")
        trn_path = 'data/%s0/train0.pt' % (dataset)
        if not os.path.exists(trn_path):
            data_prepare(manual_seed=manual_seed, dataset=dataset,
                         num_classes=self.num_classes)

        self.Trn = torch.load(trn_path)

        self.X_train = []
        self.y_train = []
        for item in self.Trn:
            data, label = item[0], item[1]
            if type(data) != np.ndarray:
                data = data.numpy()
            self.X_train.append(data.reshape(-1))
            self.y_train.append(label)
        self.X_train = np.array(self.X_train)
        self.y_train = np.array(self.y_train)

        baseline_idxs = []
        for label in np.unique(self.y_train):
            class_idx = np.where(self.y_train == label)[0]
            class_data = self.X_train[class_idx]
            intra_class_dis = [np.mean(
                np.linalg.norm(
                    class_data - class_data[data_idx], ord=1, axis=1)
            ) for data_idx in range(len(class_data))]
            baseline_idxs += class_idx[np.argsort(intra_class_dis)
                                       ][:5].tolist()
        # print('selected baseline idx: ', baseline_idxs)
        self.initial_feature_baseline = copy.deepcopy(self.X_train)[
            baseline_idxs]
        self.feature_baseline = copy.deepcopy(self.X_train)[baseline_idxs]

        self.Tst = torch.load('data/%s0/test.pt' % (dataset))
        self.complete_Tst_idx = copy.deepcopy(self.Tst.idxs)

        self.X_test = []
        self.y_test = []
        for item in self.Tst:
            data, label = item[0], item[1]
            if type(data) != np.ndarray:
                data = data.numpy()
            self.X_test.append(data.reshape(-1))
            self.y_test.append(label)
        self.X_test = np.array(self.X_test)
        self.complete_X_test = copy.deepcopy(self.X_test)
        self.y_test = np.array(self.y_test)
        self.complete_y_test = copy.deepcopy(self.y_test)

        # player setting
        self.players = [feature_idx
                        for feature_idx in range(self.Tst[0][0].reshape(-1).shape[-1])]
        # select test samples
        self.num_samples_each_class = 1  # select only one sample in each class
        self.selected_test_samples = []
        if len(self.players) > 5 and \
            False not in [len(
                np.where(self.complete_y_test == label)[0]
            ) > self.num_samples_each_class
                for label in np.unique(self.complete_y_test)]:
            for label in np.unique(self.complete_y_test):
                self.selected_test_samples += np.where(
                    self.complete_y_test == label)[0][:self.num_samples_each_class].tolist()
        else:
            self.selected_test_samples = range(len(self.complete_X_test))
        # print('selected_test_samples: ', self.selected_test_samples)

        # prepare trained model
        self.train_model()

        # used only by FIA
        self.randomSet = []
        self.testSampleFeatureSV = dict()
        self.testSampleFeatureSV_var = dict()

        self.threads = []

    def train_model(self):
        model_path = 'models/attribution_RI-%s.pt' % (
            (self.dataset if self.manual_seed == 42
             else (self.dataset+"-"+str(self.manual_seed)))
        )
        if os.path.exists(model_path):
            self.model = torch.load(model_path, map_location=self.device)
            testData = self.Tst
            print('Given model accuracy: ',
                  DNNTest(model=self.model, test_data=testData,
                          pred_print=True)
                  )
            return

        if self.dataset == 'wine':
            self.model = NN(num_feature=self.num_feature,
                            num_classes=self.num_classes)
        else:
            self.model = RegressionModel(
                num_feature=self.num_feature, num_classes=self.num_classes)
        loss_func = torch.nn.CrossEntropyLoss()
        self.model = DNNTrain(model=self.model, trn_data=self.Trn,
                              lr=self.lr, epoch=self.ep, batch_size=self.bs,
                              loss_func=loss_func)
        print('Given model accuracy: ',
              DNNTest(model=self.model, test_data=self.Tst,
                      pred_print=True)
              )
        torch.save(self.model, model_path)

    def torch_predict(self, replace_idxs, replace_val, results):
        Tst = copy.deepcopy(self.Tst)
        ori_shape = Tst.dataset.shape
        dataset = Tst.dataset.reshape((ori_shape[0], -1))
        dataset[:, replace_idxs] = torch.FloatTensor(replace_val)
        Tst.dataset = dataset.reshape(ori_shape)
        results.put(DNNTest(model=self.model,
                    test_data=Tst, metric='prediction'))

    def utility_computation(self, player_idxs):
        utility = 0.0
        replace_idxs = [feature_idx for feature_idx in range(self.X_train.shape[-1])
                        if feature_idx not in player_idxs]
        baselines = [eval(baseline)
                     for baseline in set([
                         str(tmp)
                         for tmp in self.feature_baseline[:, replace_idxs].tolist()])
                     ]
        # model testing (maybe expedited by some ML speedup functions)
        results = queue.Queue()
        for order, replace_val in enumerate(baselines):
            self.torch_predict(replace_idxs, replace_val, results)
        predictions = np.sum(list(results.queue), 0)/len(baselines)
        utility = predictions.sum()/reduce((lambda x, y: x*y), predictions.shape)

        return utility
