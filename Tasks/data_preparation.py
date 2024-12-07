import torch
import random
import os
import pandas as pd
import numpy as np

from torchvision import datasets, transforms
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, dataset, labels, original_len, idxs, attacker=False,
                 poison_labels=[], after_poison_labels=[]):
        self.dataset = dataset
        self.labels = labels
        self.idxs = list(idxs)
        self.attacker = attacker
        self.poison_labels = poison_labels
        self.after_poison_labels = after_poison_labels
        self.original_len = original_len

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        if type(self.labels) == type(None):
            data, true_label = self.dataset[self.idxs[item] %
                                            self.original_len]
        else:
            data = self.dataset[self.idxs[item] % self.original_len]
            true_label = self.labels[self.idxs[item] % self.original_len]

        if self.idxs[item] > self.original_len-1:
            # print('add noise...')
            data += torch.normal(0, 0.05, data.shape)

        if self.attacker and int(true_label) in self.poison_labels:
            label = torch.tensor(
                self.after_poison_labels[
                    self.poison_labels.index(int(true_label))
                ]
            )

        else:
            label = true_label
        return data, label, true_label


def load_AdultData(file_path):
    column_names = [
        "age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week", "native_country",
        "income"
    ]
    data = pd.read_csv(file_path, names=column_names,
                       na_values=" ?", skipinitialspace=True)
    data.head()
    data = preprocess_AdultData(data)
    return data


def preprocess_AdultData(data):
    data = data.dropna()  # 删除缺失值
    label_encoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])
    return data


def get_datasets(dataset):
    # load datasets
    if dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST(
            root='./data/mnist/',
            train=True,
            download=True,
            transform=trans_mnist
        )

        dataset_test = datasets.MNIST(
            root='./data/mnist/',
            train=False,
            download=True,
            transform=trans_mnist
        )

    elif dataset == 'cifar':
        img_size = torch.Size([3, 32, 32])

        trans_cifar10_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
        trans_cifar10_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

        dataset_train = datasets.CIFAR10(
            root='./data/cifar',
            train=True,
            download=True,
            transform=trans_cifar10_train
        )

        dataset_test = datasets.CIFAR10(
            root='./data/cifar',
            train=False,
            download=True,
            transform=trans_cifar10_val
        )
    elif dataset == 'iris':
        iris = load_iris()
        x = iris.data
        y = iris.target
        # Split dataset
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2)

        x_train = torch.FloatTensor(x_train)
        x_test = torch.FloatTensor(x_test)
        y_train = torch.LongTensor(y_train)
        y_test = torch.LongTensor(y_test)

        dataset_train = ImageDataset(x_train, y_train,
                                     len(x_train), range(len(x_train)))
        dataset_test = ImageDataset(x_test, y_test,
                                    len(x_test), range(len(x_test)))
        # img_size = x_train[0].shape
        print('iris train data shape:', x_train.shape)
        print('iris test data shape:', x_test.shape)
        print('iris labels:', set(y_train.numpy().tolist()))

    elif dataset == 'wine':
        wine = load_wine()
        x = wine.data
        y = wine.target
        # Split dataset
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2)

        x_train = torch.FloatTensor(x_train)
        x_test = torch.FloatTensor(x_test)
        y_train = torch.LongTensor(y_train)
        y_test = torch.LongTensor(y_test)

        dataset_train = ImageDataset(x_train, y_train,
                                     len(x_train), range(len(x_train)))
        dataset_test = ImageDataset(x_test, y_test,
                                    len(x_test), range(len(x_test)))
        # img_size = x_train[0].shape
        print('wine train data shape:', x_train.shape)
        print('wine test data shape:', x_test.shape)
        print('wine labels:', set(y_train.numpy().tolist()))

    else:
        exit('Error: unrecognized dataset')

    return dataset_train, dataset_test


def data_split(dataset, num_classes, data_allocation, num_trainDatasets, group_size, multiplier, data_size_group, data_size_mean):
    # load datasets
    dataset_train, dataset_test = get_datasets(dataset)
    validation_index = []  # np.random.choice(
    # len(dataset_test),int(len(dataset_test)*0.05), replace=False
    # )

    # sampling
    if data_allocation == 0 or num_trainDatasets == 1:
        data_size_group = 1
        data_size_means = [len(dataset_train)/num_trainDatasets]
        group_size = [num_trainDatasets]
    else:
        group_size = [int(tmp) for tmp in group_size.split(",")]
        multiplier = [int(tmp) for tmp in multiplier.split(",")]

        data_size_means = [data_size_mean*multiplier[gidx]
                           for gidx in range(data_size_group)]

    data_quantity = []
    for i in range(data_size_group):
        tmp = np.random.normal(data_size_means[i],
                               (0 if data_allocation == 0 or num_trainDatasets == 1
                                else data_size_means[i]/4),
                               group_size[i])
        tmp2 = []
        small_index = np.where(tmp <= data_size_means[i])[0]
        if len(small_index) >= group_size[i]/2:
            tmp2 += list(tmp[small_index][:int(group_size[i]/2)])
            tmp2 += list(2*data_size_means[i] -
                         tmp[small_index][:group_size[i]-int(group_size[i]/2)])
        else:
            large_index = np.where(tmp >= data_size_means[i])[0]
            tmp2 += list(tmp[large_index][:int(group_size[i]/2)])
            tmp2 += list(2*data_size_means[i] -
                         tmp[large_index][:group_size[i]-int(group_size[i]/2)])
        # tmp2 = tmp2[:group_size[i]]
        if len(tmp2) < group_size[i]:
            tmp2 += tmp2 + tmp2[int(group_size[i]/2):
                                int(group_size[i]/2)+(group_size[i]-len(tmp2))]
        data_quantity += tmp2
    data_quantity = np.array([(int(np.round(i)) if int(np.round(i)) >= 2 else 2)
                              for i in data_quantity])
    data_quantity = sorted(data_quantity)
    print(data_quantity)
    if len(group_size) <= 1:
        data_idx = list(range(sum(data_quantity)))
        # print(data_idx)
        np.random.shuffle(data_idx)
        workers_idxs = [[] for _ in range(num_trainDatasets)]
        for idx in range(num_trainDatasets):
            print('sampling worker %s...' % idx)
            workers_idxs[idx] = np.random.choice(data_idx,
                                                 data_quantity[idx], replace=False)
            data_idx = list(set(data_idx)-set(workers_idxs[idx]))
            np.random.shuffle(data_idx)
    else:
        try:
            idxs_labels = np.array(dataset_train.train_labels)
        except:
            try:
                idxs_labels = np.array(dataset_train.targets)
            except:
                idxs_labels = np.array(dataset_train.labels)

        class_num = dict([(c, 0) for c in range(num_classes)])
        worker_classes = dict()
        for idx in range(num_trainDatasets):
            worker_classes[idx] = range(num_classes)
            for tmp, c in enumerate(worker_classes[idx]):
                if tmp == len(worker_classes[idx])-1:
                    class_num[c] += data_quantity[idx] - \
                        int(data_quantity[idx]/len(worker_classes[idx])
                            )*(len(worker_classes[idx])-1)
                else:
                    class_num[c] += int(data_quantity[idx] /
                                        len(worker_classes[idx]))

        class_indexes = dict()
        for c, num in class_num.items():
            original_index = list(np.where(idxs_labels == c)[0])
            appended_index = []
            count = 0
            while len(appended_index) < num:
                appended_index += [tmp+count *
                                   len(idxs_labels) for tmp in original_index]
                count += 1
            np.random.shuffle(appended_index)
            class_indexes[c] = appended_index

        workers_idxs = [[] for _ in range(num_trainDatasets)]
        for idx in range(num_trainDatasets):
            print('sampling worker %s...' % idx)
            workers_idxs[idx] = []
            for tmp, c in enumerate(worker_classes[idx]):
                if tmp == len(worker_classes[idx])-1:
                    sampled_idx = list(np.random.choice(
                        class_indexes[c],
                        data_quantity[idx] -
                        int(data_quantity[idx]/len(worker_classes[idx])
                            )*(len(worker_classes[idx])-1),
                        replace=False))
                else:
                    sampled_idx = list(np.random.choice(
                        class_indexes[c],
                        int(data_quantity[idx]/len(worker_classes[idx])),
                        replace=False))
                workers_idxs[idx] += sampled_idx
                class_indexes[c] = list(set(class_indexes[c])-set(sampled_idx))
                np.random.shuffle(class_indexes[c])
            np.random.shuffle(workers_idxs[idx])
            print(data_quantity[idx], len(workers_idxs[idx]),
                  worker_classes[idx], set([idxs_labels[tmp % len(idxs_labels)] for tmp in workers_idxs[idx]]))

    dict_workers = {i: workers_idxs[i] for i in range(len(workers_idxs))}
    x = []
    combine = []
    for i in dict_workers.values():
        x.append(len(i))
        combine.append(len(i))
    print('train data partition')
    print('sum:', np.sum(np.array(x)))
    print('mean:', np.mean(np.array(x)))
    print('std:', np.std(np.array(x)))
    print('max:', max(np.array(x)))
    print('min:', min(np.array(x)))
    return dataset_train, dataset_test, validation_index, dict_workers


def init_random_seed(manual_seed):
    seed = None
    if manual_seed is None:
        seed = random.randint(1, 10000)
    else:
        seed = manual_seed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.backends.cudnn.deterministic = True
    # if False, sacrifice the computation efficiency
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def data_prepare(manual_seed, dataset, num_classes, data_allocation=0, num_trainDatasets=1,
                 group_size='1', multiplier='1', data_size_group=1, data_size_mean=100.0):
    init_random_seed(manual_seed)

    # load dataset and split workers
    dataset_train, dataset_test, validation_index, dict_workers = data_split(
        dataset, num_classes, data_allocation, num_trainDatasets, group_size, multiplier, data_size_group, data_size_mean)
    trn_len = len(dataset_train)
    tst_len = len(dataset_test)

    if os.path.exists('data/%s%s/' % (dataset, data_allocation)) == False:
        os.makedirs('data/%s%s/' % (dataset, data_allocation))
    for idx in range(num_trainDatasets):
        filename = 'data/%s%s/train%s.pt' % (dataset, data_allocation, idx)
        if type(dataset_train) == ImageDataset:
            trainset = dataset_train
            trainset.idxs = dict_workers[idx]
        else:
            trainset = ImageDataset(
                dataset_train, None, trn_len, dict_workers[idx],
            )
        # else:
        #    trainset = DatasetSplit(dataset, data_idxs)

        torch.save(trainset, filename)
        print('Trainset %s: data_size %s %s...' % (
            idx, len(dict_workers[idx]), len(trainset)))

    test_dataset_path = 'data/%s%s/test.pt' % (dataset, data_allocation)
    test_idxs = list(set(range(len(dataset_test)))-set(validation_index))
    if type(dataset_train) == ImageDataset:
        testset = dataset_test
        testset.idxs = test_idxs
    else:
        testset = ImageDataset(
            dataset_test, None, tst_len, test_idxs
        )
    torch.save(testset, test_dataset_path)
    print('test data_size %s...' % len(testset))
