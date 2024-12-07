import torch
import copy
import numpy as np
import torch.nn.functional as F

from torch.utils.data import DataLoader
from .nets import CNN, RegressionModel, LinearAttackModel, CNNCifar, NN

TEST_BS = 128


def find_free_device():
    """
    Find and return the index of the most free GPU.
    Return -1 if no GPU is available.
    """
    free_gpu_index = -1
    free_gpu_memory = -1

    for gpu_index in range(torch.cuda.device_count()):
        gpu_memory_allocated = torch.cuda.memory_allocated(device=gpu_index)
        if gpu_memory_allocated == 0:
            free_gpu_index = gpu_index
            break
        elif free_gpu_memory < 0 or gpu_memory_allocated < free_gpu_memory:
            free_gpu_index = gpu_index
            free_gpu_memory = gpu_memory_allocated

    return torch.device(
        'cuda:{}'.format(free_gpu_index)
        if free_gpu_index != -1 and torch.cuda.is_available() else 'cpu'
    )


def DNNTrain(model, trn_data, lr, epoch=1, batch_size=128, loss_func=None,
             momentum=0, weight_decay=0, max_norm=5,
             validation_metric='tst_accuracy'):
    model = copy.deepcopy(model)
    if type(model) == CNN or type(model) == CNNCifar or type(model) == NN or \
            type(model) == RegressionModel or type(model) == LinearAttackModel:
        device = find_free_device()
        model = model.to(device)

        print_flag = False
        if len(trn_data)/batch_size > 50 or epoch > 100:
            print_flag = True
            val_data = copy.deepcopy(trn_data)
            trn_data_idxs = trn_data.idxs
            val_data_idxs = np.random.choice(trn_data_idxs,
                                             int(len(trn_data_idxs)/5))
            val_data.idxs = val_data_idxs

        ldr_train = DataLoader(trn_data,
                               batch_size=batch_size,
                               shuffle=True)
        num_batch = len(ldr_train)
        model.train()
        for (n, p) in model.named_parameters():
            p.requires_grad = True
        if type(model) == NN:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        else:
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay)

        convergeFlag = False
        val_results = []
        for iter_ in range(epoch):
            if convergeFlag:
                break
            for batch_idx, batch in enumerate(ldr_train):
                if print_flag:
                    if batch_idx == num_batch-1:
                        val_data.idxs = np.random.choice(
                            trn_data_idxs, int(len(trn_data_idxs)/5))
                        val_results.append(
                            DNNTest(model, val_data, metric=validation_metric))
                        if len(val_results) > 3 and\
                                np.max(val_results[-3:]) < 0.1*val_results[0]:
                            convergeFlag = True

                optimizer.zero_grad()
                model.zero_grad()

                data = batch[0].to(device)
                labels = batch[1].to(device)
                # forward
                net_outputs = model(data)
                # loss
                loss = loss_func(net_outputs, labels)
                loss.backward()
                if max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        parameters=model.parameters(),
                        max_norm=max_norm,
                        norm_type=2)
                optimizer.step()
        del ldr_train
    else:
        pass

    torch.cuda.empty_cache()
    return model


def DNNTest(model, test_data,
            metric='tst_accuracy',
            record_skippable_sample=None, pred_print=False):
    test_bs = TEST_BS
    # model testing (maybe expedited by some ML speedup functions)
    if type(model) == CNN or type(model) == CNNCifar or type(model) == NN or\
            type(model) == RegressionModel or type(model) == LinearAttackModel:

        param = list(model.parameters())[0]
        if 'cuda' in str(param.device):
            device = param.device

        else:
            device = find_free_device()
        del param

        ldr_eval = DataLoader(test_data,
                              batch_size=test_bs,
                              shuffle=False)
        model = model.to(device)
        model.eval()
        data_size = 0
        batch_loss = []
        correct = 0
        predictions = []
        targets = []
        for batch_idx, batch in enumerate(ldr_eval):
            data = batch[0].to(device)
            labels = batch[1].to(device)
            # forward
            outputs = model(data)
            predictions.append(outputs.data.cpu().numpy())
            targets.append(labels.data.cpu().numpy())

            # record_skippable_sample
            if type(record_skippable_sample) != type(None):
                (skippableTestSample, playerIdx) = record_skippable_sample
                y_pred = outputs.data.max(1, keepdim=True)[1]
                for data_idx, (l, pl) in enumerate(zip(labels, y_pred)):
                    if int(l) == int(pl):
                        skippableTestSample[playerIdx].add(data_idx+data_size)
            # metric
            if metric == 'tst_accuracy':
                y_pred = outputs.data.max(1, keepdim=True)[1]
                correct += y_pred.eq(labels.data.view_as(y_pred)
                                     ).long().cpu().sum()

            elif metric == 'tst_loss':
                loss = F.cross_entropy(outputs, labels, reduction='sum').item()
                batch_loss.append(loss)

            data_size += len(labels)

        if metric == 'tst_accuracy':
            utility = float(correct.item() / data_size)*100
        elif metric == 'tst_loss':
            utility = sum(batch_loss) / data_size
        elif metric == 'prediction':
            utility = np.concatenate(predictions)
        elif metric == 'tst_MAE':
            if pred_print:
                print("predictions: ", predictions)
                print("targets: ", targets)
            utility = F.l1_loss(torch.tensor(np.concatenate(predictions)),
                                torch.tensor(np.concatenate(targets)))

    return utility
