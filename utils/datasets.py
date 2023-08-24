"""
This module contains various dataset type classes.

"""

import pickle

import numpy as np
import torch
from torch.utils.data import Dataset


class DatasetSequenceOfSnapshots(Dataset):
    def __init__(self, dataset_address, dataset=None):
        if dataset is not None:
            self.dataset = dataset
        else:
            with open(dataset_address, "rb") as f:
                self.dataset = pickle.load(f)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        sample = self.dataset[item]
        features_array = sample['array_output']
        features_real = np.real(features_array)
        features_imag = np.imag(features_array)
        features = torch.from_numpy(np.transpose(np.concatenate((features_real, features_imag), axis=0))).float()
        targets = torch.FloatTensor(sample['target_vector'])
        return features, targets


def sequence_of_snapshots_fixed_n_collate(batch):
    data = [torch.unsqueeze(item[0], 1) for item in batch]
    target = [item[1] for item in batch]
    data = torch.cat(data, 1)
    target = torch.vstack(target)
    return [data, target]


class DatasetSequenceOfSnapshotsSortedNumSnapshots(Dataset):
    def __init__(self, dataset_address, dataset=None):
        if dataset is not None:
            self.dataset = dataset
        else:
            with open(dataset_address, "rb") as f:
                self.dataset = pickle.load(f)
        self.dataset = list(sorted(self.dataset, key=lambda x: x['sample_info']['N'], reverse=True))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        sample = self.dataset[item]
        features_array = sample['array_output']
        features_real = np.real(features_array)
        features_imag = np.imag(features_array)
        features = torch.from_numpy(np.transpose(np.concatenate((features_real, features_imag), axis=0))).float()
        targets = torch.FloatTensor(sample['target_vector'])
        return features, targets


def sequence_of_snapshots_variable_n_collate(batch):
    data = [torch.unsqueeze(item[0], 1) for item in batch]
    max_seq_len = 0
    for item in data:
        if item.shape[0] > max_seq_len:
            max_seq_len = item.shape[0]
    for i in range(len(data)):
        data[i] = padding_data(data[i], max_seq_len)
    target = [item[1] for item in batch]
    data = torch.cat(data, 1)
    target = torch.vstack(target)
    return [data, target]


class DatasetSequenceOfAntennas(Dataset):
    def __init__(self, dataset_address, dataset=None):
        if dataset is not None:
            self.dataset = dataset
        else:
            with open(dataset_address, "rb") as f:
                self.dataset = pickle.load(f)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        sample = self.dataset[item]
        features_array = sample['array_output']
        features_real = np.real(features_array)
        features_imag = np.imag(features_array)
        features = torch.from_numpy(np.concatenate((features_real, features_imag), axis=0)).float()
        targets = torch.FloatTensor(sample['target_vector'])
        return features, targets


def sequence_of_antennas_collate(batch):
    data = [torch.unsqueeze(item[0], 1) for item in batch]
    target = [item[1] for item in batch]
    data = torch.cat(data, 1)
    target = torch.vstack(target)
    return [data, target]


class DatasetSequenceOfCovMatrixRows(Dataset):
    def __init__(self, dataset_address, dataset=None):
        if dataset is not None:
            self.dataset = dataset
        else:
            with open(dataset_address, "rb") as f:
                self.dataset = pickle.load(f)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        sample = self.dataset[item]
        features_array = sample['array_output']
        features_cov = np.cov(features_array, bias=True)
        features_cov_real = np.real(features_cov)
        features_cov_imag = np.imag(features_cov)
        features = torch.from_numpy(np.concatenate((features_cov_real, features_cov_imag), axis=1)).float()
        targets = torch.FloatTensor(sample['target_vector'])
        return features, targets


def sequence_of_cov_matrix_rows_collate(batch):
    data = [torch.unsqueeze(item[0], 1) for item in batch]
    target = [item[1] for item in batch]
    data = torch.cat(data, 1)
    target = torch.vstack(target)
    return [data, target]


class DatasetSequenceOfSnapshotsForSeq2Seq(Dataset):
    def __init__(self, dataset_address, trg_len, dataset=None):
        if dataset is not None:
            self.dataset = dataset
        else:
            with open(dataset_address, "rb") as f:
                self.dataset = pickle.load(f)
        self.trg_len = trg_len
        self.sos_index = 0
        self.eos_index = 1

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        sample = self.dataset[item]
        features_array = sample['array_output']
        features_real = np.real(features_array)
        features_imag = np.imag(features_array)
        features = torch.from_numpy(np.transpose(np.concatenate((features_real, features_imag), axis=0))).float()
        targets = torch.LongTensor(sample['target_sequence'])
        targets = targets + 2
        targets = torch.hstack((torch.LongTensor([self.sos_index]), targets,
                                torch.LongTensor([self.eos_index] * (self.trg_len - len(targets) - 1))))
        return features, targets


def sequence_of_snapshots_fixed_n_for_seq2seq_collate(batch):
    data = [torch.unsqueeze(item[0], 1) for item in batch]
    target = [item[1] for item in batch]
    data = torch.cat(data, 1)
    target = torch.vstack(target)
    target = torch.transpose(target, 0, 1)
    return [data, target]


class DatasetSequenceOfSnapshotsForSeq2SeqSortedNumSnapshots(Dataset):
    def __init__(self, dataset_address, trg_len, dataset=None):
        if dataset is not None:
            self.dataset = dataset
        else:
            with open(dataset_address, "rb") as f:
                self.dataset = pickle.load(f)
        self.dataset = list(sorted(self.dataset, key=lambda x: x['sample_info']['N'], reverse=True))
        self.trg_len = trg_len
        self.sos_index = 0
        self.eos_index = 1

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        sample = self.dataset[item]
        features_array = sample['array_output']
        features_real = np.real(features_array)
        features_imag = np.imag(features_array)
        features = torch.from_numpy(np.transpose(np.concatenate((features_real, features_imag), axis=0))).float()
        targets = torch.LongTensor(sample['target_sequence'])
        targets = targets + 2
        targets = torch.hstack((torch.LongTensor([self.sos_index]), targets,
                                torch.LongTensor([self.eos_index] * (self.trg_len - len(targets) - 1))))
        return features, targets


def sequence_of_snapshots_variable_n_for_seq2seq_collate(batch):
    data = [torch.unsqueeze(item[0], 1) for item in batch]
    max_seq_len = 0
    for item in data:
        if item.shape[0] > max_seq_len:
            max_seq_len = item.shape[0]
    for i in range(len(data)):
        data[i] = padding_data(data[i], max_seq_len)
    target = [item[1] for item in batch]
    data = torch.cat(data, 1)
    target = torch.vstack(target)
    target = torch.transpose(target, 0, 1)
    return [data, target]


class DatasetSequenceOfCovMatrixRowsForSeq2Seq(Dataset):
    def __init__(self, dataset_address, trg_len, dataset=None):
        if dataset is not None:
            self.dataset = dataset
        else:
            with open(dataset_address, "rb") as f:
                self.dataset = pickle.load(f)
        self.trg_len = trg_len
        self.sos_index = 0
        self.eos_index = 1

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        sample = self.dataset[item]
        features_array = sample['array_output']
        features_cov = np.cov(features_array, bias=True)
        features_cov_real = np.real(features_cov)
        features_cov_imag = np.imag(features_cov)
        features = torch.from_numpy(np.concatenate((features_cov_real, features_cov_imag), axis=1)).float()
        targets = torch.LongTensor(sample['target_sequence'])
        targets = targets + 2
        targets = torch.hstack((torch.LongTensor([self.sos_index]), targets,
                                torch.LongTensor([self.eos_index] * (self.trg_len - len(targets) - 1))))
        return features, targets


def sequence_of_cov_matrix_rows_for_seq2seq_collate(batch):
    data = [torch.unsqueeze(item[0], 1) for item in batch]
    target = [item[1] for item in batch]
    data = torch.cat(data, 1)
    target = torch.vstack(target)
    target = torch.transpose(target, 0, 1)
    return [data, target]


def padding_data(data, max_seq_len):
    if data.shape[0] < max_seq_len:
        pad_shape = list(data.shape)
        pad_shape[0] = max_seq_len - data.shape[0]
        data_pad = torch.zeros(pad_shape)
        data = torch.concat((data, data_pad), dim=0)
    return data
