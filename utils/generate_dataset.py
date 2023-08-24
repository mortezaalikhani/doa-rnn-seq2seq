"""
This module contains a class that generates customized doa estimation datasets.

"""

import argparse
import math
import os
import pickle
import random

import numpy as np
from scipy import linalg
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class GenerateDatasetULA():
    """
    This class generates customized doa estimation datasets.

    """

    @staticmethod
    def merge_datasets(output_path, dataset_list, display=False):
        os.makedirs(output_path, exist_ok=True)
        train_dataset_final = []
        validation_dataset_final = []
        test_dataset_final = []
        info_dataset_final = {"M": [], "N": [], "K": [], "d": [], "wavelength": [],
                              "doa_min": [], "doa_max": [], "num_rep": [], "step": [],
                              "max_num_samples": [], "coherent_signals_max_snr_diff": [],
                              "doa_delta_list": [], "snr_diff_list": [],
                              "num_coherent_signal": [], "imperfections": []}
        for dataset in tqdm(dataset_list, disable=not display):
            with open(os.path.join(dataset, "train.pkl"), "rb") as f:
                train_dataset = pickle.load(f)
            train_dataset_final.extend(train_dataset)
            with open(os.path.join(dataset, "validation.pkl"), "rb") as f:
                validation_dataset = pickle.load(f)
            validation_dataset_final.extend(validation_dataset)
            with open(os.path.join(dataset, "test.pkl"), "rb") as f:
                test_dataset = pickle.load(f)
            test_dataset_final.extend(test_dataset)
            with open(os.path.join(dataset, "info.pkl"), "rb") as f:
                info_dataset = pickle.load(f)
            for key, value in info_dataset_final.items():
                value.extend(info_dataset[key])
        with open(os.path.join(output_path, "train.pkl"), "wb") as f:
            pickle.dump(train_dataset_final, f)
        with open(os.path.join(output_path, "validation.pkl"), "wb") as f:
            pickle.dump(validation_dataset_final, f)
        with open(os.path.join(output_path, "test.pkl"), "wb") as f:
            pickle.dump(test_dataset_final, f)
        with open(os.path.join(output_path, "info.pkl"), "wb") as f:
            pickle.dump(info_dataset_final, f)

    def generate_dataset_const_n_k_snr(self, output_path, M, N, K, SNR, d, wavelength, doa_min=-60, doa_max=60,
                                       num_rep=1, step=1,
                                       max_num_samples=1000000, coherent_signals_max_snr_diff=4, doa_delta_list=None,
                                       snr_diff_list=None, num_coherent_signal=0, gain_inconsistence=0,
                                       phase_inconsistence=0,
                                       sensor_position_error=0, mutual_coupling=0,
                                       gain_inconsistence_array=None, phase_inconsistence_array=None,
                                       sensor_position_error_array=None, mutual_coupling_array=None, display=False,
                                       train_ratio=0.8,
                                       test_ratio=0.1):
        assert 0 <= train_ratio and 0 <= test_ratio and train_ratio + test_ratio <= 1
        os.makedirs(output_path, exist_ok=True)
        dataset = self.generate_samples(M, N, K, SNR, d, wavelength, doa_min, doa_max, num_rep, step, max_num_samples,
                                        coherent_signals_max_snr_diff, doa_delta_list, snr_diff_list,
                                        num_coherent_signal, display)
        if gain_inconsistence != 0 or phase_inconsistence != 0 or sensor_position_error != 0 or mutual_coupling != 0:
            self.add_imperfections_to_samples(dataset, gain_inconsistence, phase_inconsistence,
                                              sensor_position_error, mutual_coupling,
                                              gain_inconsistence_array, phase_inconsistence_array,
                                              sensor_position_error_array, mutual_coupling_array)
            info = {"M": [M], "N": [N], "K": [K], "d": [d], "wavelength": [wavelength],
                    "doa_min": [doa_min], "doa_max": [doa_max], "num_rep": [num_rep], "step": [step],
                    "max_num_samples": [max_num_samples],
                    "coherent_signals_max_snr_diff": [coherent_signals_max_snr_diff],
                    "doa_delta_list": [doa_delta_list], "snr_diff_list": [snr_diff_list],
                    "num_coherent_signal": [num_coherent_signal], "imperfections": [dataset["imperfections"]]}
        else:
            info = {"M": [M], "N": [N], "K": [K], "d": [d], "wavelength": [wavelength],
                    "doa_min": [doa_min], "doa_max": [doa_max], "num_rep": [num_rep], "step": [step],
                    "max_num_samples": [max_num_samples],
                    "coherent_signals_max_snr_diff": [coherent_signals_max_snr_diff],
                    "doa_delta_list": [doa_delta_list], "snr_diff_list": [snr_diff_list],
                    "num_coherent_signal": [num_coherent_signal], "imperfections": [[]]}
        with open(os.path.join(output_path, "info.pkl"), "wb") as f:
            pickle.dump(info, f)
        samples_list = dataset["samples_list"]
        dataset_modified = []
        for sample in samples_list:
            doa_list = sample["doa_list"]
            target_sequence = [doa - doa_min for doa in doa_list]
            target_vector = [0] * (doa_max - doa_min)
            for i in target_sequence:
                target_vector[i] = 1
            sample_info = {"doa_list": sample["doa_list"], "snr_list": sample["snr_list"],
                           "snr_diff": sample["snr_diff"],
                           "coherent_coefs_list": sample["coherent_coefs_list"], "M": M, "N": N, "K": K, "d": d,
                           "wavelength": wavelength,
                           "num_coherent_signal": num_coherent_signal, "imperfections": info["imperfections"][0]}
            dataset_modified.append({"array_output": sample["array_output"], "target_sequence": target_sequence,
                                     "target_vector": target_vector, "sample_info": sample_info})
        train_validation_dataset, test_dataset = train_test_split(dataset_modified, test_size=test_ratio)
        train_dataset, validation_dataset = train_test_split(train_validation_dataset,
                                                             test_size=(1 - train_ratio - test_ratio) / (
                                                                     1 - test_ratio))
        with open(os.path.join(output_path, "train.pkl"), "wb") as f:
            pickle.dump(train_dataset, f)
        with open(os.path.join(output_path, "validation.pkl"), "wb") as f:
            pickle.dump(validation_dataset, f)
        with open(os.path.join(output_path, "test.pkl"), "wb") as f:
            pickle.dump(test_dataset, f)

    def generate_samples(self, M, N, K, SNR, d, wavelength, doa_min=-60, doa_max=60, num_rep=1, step=1,
                         max_num_samples=1000000, coherent_signals_max_snr_diff=4, doa_delta_list=None,
                         snr_diff_list=None, num_coherent_signal=0, display=False):
        assert K >= 1
        assert 0 <= num_coherent_signal <= K and num_coherent_signal != 1
        assert 0 < num_rep < 1 or (num_rep >= 1 and num_rep - int(num_rep) == 0)
        samples_list = []
        if K == 1:
            sample_space = np.array(range(doa_min, doa_max, step))
            sample_space = np.expand_dims(sample_space, axis=-1)
            if num_rep >= 1:
                samples_doa = sample_space
                if num_rep > 1:
                    for rep_index in range(num_rep - 1):
                        samples_doa = np.vstack([samples_doa, sample_space])
            else:
                np.random.shuffle(sample_space)
                num_samples = int(np.ceil(num_rep * len(sample_space)))
                samples_doa = sample_space[0:num_samples]
            for sample_doa in tqdm(samples_doa, disable=not display):
                doa = int(sample_doa[0])
                add_noise = np.random.randn(M, N) + 1j * np.random.randn(M, N)
                array_signal = self.generate_single_signal_array(doa, SNR, M, N, d, wavelength)
                array_signal_list = [array_signal]
                snr_list = [SNR]
                snr_diff = 0
                coherent_coefs_list = [(0, 0)]
                array_output_nf = array_signal + 0 * add_noise  # noise-free output
                array_output = array_signal + 1 * add_noise
                doa_list = [doa]
                samples_list.append(
                    {"doa_list": doa_list, "array_output": array_output, "array_output_nf": array_output_nf,
                     "array_signal_list": array_signal_list, "snr_list": snr_list, "snr_diff": snr_diff,
                     "coherent_coefs_list": coherent_coefs_list})
        elif K == 2:
            if snr_diff_list is None:
                snr_diffs = [0]
            else:
                snr_diffs = snr_diff_list
            if doa_delta_list is None:
                doa_deltas = list(range(1, doa_max - doa_min))
            else:
                doa_deltas = doa_delta_list
            sample_space = []
            for doa_delta in doa_deltas:
                for first_doa in range(doa_min, doa_max - doa_delta, step):
                    second_doa = first_doa + doa_delta
                    sample_space.append([first_doa, second_doa])
            sample_space = np.array(sample_space)
            if num_rep >= 1:
                samples_doa = sample_space
                if num_rep > 1:
                    for rep_index in range(num_rep - 1):
                        samples_doa = np.vstack([samples_doa, sample_space])
            else:
                np.random.shuffle(sample_space)
                num_samples = int(np.ceil(num_rep * len(sample_space)))
                samples_doa = sample_space[0:num_samples]
            for sample_doa in tqdm(samples_doa, disable=not display):
                doa_list = [int(doa) for doa in sample_doa]
                if num_coherent_signal == 0:
                    for snr_diff in snr_diffs:
                        add_noise = np.random.randn(M, N) + 1j * np.random.randn(M, N)
                        snr_list = [SNR, SNR - snr_diff]
                        random.shuffle(snr_list)
                        array_signal = 0
                        array_signal_list = []
                        for i in range(K):
                            array_signal_i = self.generate_single_signal_array(doa_list[i], snr_list[i], M, N, d,
                                                                               wavelength)
                            array_signal_list.append(array_signal_i)
                            array_signal += array_signal_i
                        coherent_coefs_list = [(0, 0), (0, 0)]
                        array_output_nf = array_signal + 0 * add_noise  # noise-free output
                        array_output = array_signal + 1 * add_noise
                        samples_list.append(
                            {"doa_list": doa_list, "array_output": array_output, "array_output_nf": array_output_nf,
                             "array_signal_list": array_signal_list, "snr_list": snr_list, "snr_diff": snr_diff,
                             "coherent_coefs_list": coherent_coefs_list})
                else:
                    add_noise = np.random.randn(M, N) + 1j * np.random.randn(M, N)
                    snr_diff = coherent_signals_max_snr_diff * np.random.rand()
                    a = 10 ** (snr_diff / 20)
                    b = 2 * np.pi * np.random.rand()
                    array_signal_list = []
                    if np.random.rand() <= 0.5:
                        snr_list = [SNR, SNR - snr_diff]
                        coherent_coefs_list = [(1, 0), (1 / a, b)]
                        signal = (10 ** (SNR / 20)) * (np.random.randn(1, N) + 1j * np.random.randn(1, N))
                        array_signal_i = self.generate_single_signal_array(doa_list[0], SNR, M, N, d, wavelength,
                                                                           signal)
                        array_signal_list.append(array_signal_i)
                        array_signal = array_signal_i
                        signal = (1 / a) * np.exp(1j * b) * signal
                        array_signal_i = self.generate_single_signal_array(doa_list[1], SNR - snr_diff, M, N, d,
                                                                           wavelength,
                                                                           signal)
                        array_signal_list.append(array_signal_i)
                        array_signal += array_signal_i
                    else:
                        snr_list = [SNR - snr_diff, SNR]
                        coherent_coefs_list = [(1 / a, b), (1, 0)]
                        signal = (10 ** (SNR / 20)) * (np.random.randn(1, N) + 1j * np.random.randn(1, N))
                        array_signal_i = self.generate_single_signal_array(doa_list[1], SNR, M, N, d, wavelength,
                                                                           signal)
                        array_signal_list.append(array_signal_i)
                        array_signal = array_signal_i
                        signal = (1 / a) * np.exp(1j * b) * signal
                        array_signal_i = self.generate_single_signal_array(doa_list[0], SNR - snr_diff, M, N, d,
                                                                           wavelength,
                                                                           signal)
                        array_signal_list.append(0)
                        array_signal_list[1] = array_signal_list[0]
                        array_signal_list[0] = array_signal_i
                        array_signal += array_signal_i
                    array_output_nf = array_signal + 0 * add_noise  # noise-free output
                    array_output = array_signal + 1 * add_noise
                    samples_list.append(
                        {"doa_list": doa_list, "array_output": array_output, "array_output_nf": array_output_nf,
                         "array_signal_list": array_signal_list, "snr_list": snr_list, "snr_diff": snr_diff,
                         "coherent_coefs_list": coherent_coefs_list})
        else:
            if snr_diff_list is None:
                snr_diffs = [0]
            else:
                snr_diffs = snr_diff_list
            if num_rep >= 1:
                sample_space = self.get_sample_space(doa_min, doa_max, K)
                sample_space = np.array(sample_space)
                samples_doa = sample_space
                if num_rep > 1:
                    for rep_index in range(num_rep - 1):
                        samples_doa = np.vstack([samples_doa, sample_space])
            else:
                num_samples = int(np.ceil(num_rep * max_num_samples))
                samples_doa = []
                for i in range(num_samples):
                    samples_doa.append(
                        np.sort(np.random.choice(range(doa_min, doa_max), K, replace=False), axis=-1, kind=None,
                                order=None))
                samples_doa = np.array(samples_doa)
            for sample_doa in tqdm(samples_doa, disable=not display):
                doa_list = [int(doa) for doa in sample_doa]
                if num_coherent_signal == 0:
                    for snr_diff in snr_diffs:
                        add_noise = np.random.randn(M, N) + 1j * np.random.randn(M, N)
                        snr_list = [SNR - i * snr_diff for i in range(K)]
                        random.shuffle(snr_list)
                        array_signal = 0
                        array_signal_list = []
                        for i in range(K):
                            array_signal_i = self.generate_single_signal_array(doa_list[i], snr_list[i], M, N, d,
                                                                               wavelength)
                            array_signal_list.append(array_signal_i)
                            array_signal += array_signal_i
                        coherent_coefs_list = [(0, 0)] * K
                        array_output_nf = array_signal + 0 * add_noise  # noise-free output
                        array_output = array_signal + 1 * add_noise
                        samples_list.append(
                            {"doa_list": doa_list, "array_output": array_output, "array_output_nf": array_output_nf,
                             "array_signal_list": array_signal_list, "snr_list": snr_list, "snr_diff": snr_diff,
                             "coherent_coefs_list": coherent_coefs_list})
                else:
                    add_noise = np.random.randn(M, N) + 1j * np.random.randn(M, N)
                    signal_type_list = [0] + (num_coherent_signal - 1) * [1] + (K - num_coherent_signal) * [2]
                    random.shuffle(signal_type_list)
                    main_coherent_signal = (10 ** (SNR / 20)) * (np.random.randn(1, N) + 1j * np.random.randn(1, N))
                    coherent_coefs_list = []
                    snr_list = []
                    array_signal = 0
                    array_signal_list = []
                    for i, doa in enumerate(doa_list):
                        if signal_type_list[i] == 0:
                            array_signal_i = self.generate_single_signal_array(doa, SNR, M, N, d,
                                                                               wavelength,
                                                                               main_coherent_signal)
                            array_signal_list.append(array_signal_i)
                            array_signal += array_signal_i
                            coherent_coefs_list.append((1, 0))
                            snr_list.append(SNR)
                        elif signal_type_list[i] == 1:
                            snr_diff = coherent_signals_max_snr_diff * np.random.rand()
                            a = 10 ** (snr_diff / 20)
                            b = 2 * np.pi * np.random.rand()
                            signal = (1 / a) * np.exp(1j * b) * main_coherent_signal
                            array_signal_i = self.generate_single_signal_array(doa, SNR - snr_diff, M, N, d,
                                                                               wavelength,
                                                                               signal)
                            array_signal_list.append(array_signal_i)
                            array_signal += array_signal_i
                            coherent_coefs_list.append((1 / a, b))
                            snr_list.append(SNR - snr_diff)
                        else:
                            array_signal_i = self.generate_single_signal_array(doa, SNR, M, N, d,
                                                                               wavelength)
                            array_signal_list.append(array_signal_i)
                            array_signal += array_signal_i
                            coherent_coefs_list.append((0, 0))
                            snr_list.append(SNR)
                    array_output_nf = array_signal + 0 * add_noise  # noise-free output
                    array_output = array_signal + 1 * add_noise
                    snr_diff = 0
                    samples_list.append(
                        {"doa_list": doa_list, "array_output": array_output, "array_output_nf": array_output_nf,
                         "array_signal_list": array_signal_list, "snr_list": snr_list, "snr_diff": snr_diff,
                         "coherent_coefs_list": coherent_coefs_list})
        dataset = {"samples_list": samples_list, "M": M, "N": N, "K": K, "d": d, "wavelength": wavelength,
                   "doa_min": doa_min, "doa_max": doa_max, "num_rep": num_rep, "step": step,
                   "max_num_samples": max_num_samples, "coherent_signals_max_snr_diff": coherent_signals_max_snr_diff,
                   "doa_delta_list": doa_delta_list, "snr_diff_list": snr_diff_list,
                   "num_coherent_signal": num_coherent_signal, "imperfections": []}
        return dataset

    @staticmethod
    def generate_single_signal_array(doa, snr, M, N, d, wavelength, input_signal=None):
        if input_signal is None:
            signal = (10 ** (snr / 20)) * (np.random.randn(1, N) + 1j * np.random.randn(1, N))
        else:
            signal = input_signal
        phase_shift_unit = 2 * np.pi * d / wavelength * np.sin((doa / 180) * np.pi)
        a_ = np.cos(np.array(range(M)) * phase_shift_unit) + 1j * np.sin(
            np.array(range(M)) * phase_shift_unit)
        a = np.expand_dims(a_, axis=-1)
        array_signal = np.matmul(a, signal)
        return array_signal

    def get_sample_space(self, doa_min, doa_max, K):
        if K == 1:
            sample_space = [[doa] for doa in range(doa_min, doa_max)]
        else:
            sample_space = []
            for first_doa in range(doa_min, doa_max - K + 1):
                sample_space_temp = self.get_sample_space(first_doa + 1, doa_max, K - 1)
                sample_space_temp = [[first_doa] + sample for sample in sample_space_temp]
                sample_space.extend(sample_space_temp)
        return sample_space

    def generate_one_sample(self, M, N, doa_list, snr_list, d, wavelength, doa_min=-60, doa_max=60,
                            coherency_list=None):
        K = len(doa_list)
        if coherency_list is None:
            add_noise = np.random.randn(M, N) + 1j * np.random.randn(M, N)
            array_signal = 0
            array_signal_list = []
            for i in range(K):
                array_signal_i = self.generate_single_signal_array(doa_list[i], snr_list[i], M, N, d,
                                                                   wavelength)
                array_signal_list.append(array_signal_i)
                array_signal += array_signal_i
            coherent_coefs_list = [(0, 0)] * K
            array_output_nf = array_signal + 0 * add_noise  # noise-free output
            array_output = array_signal + 1 * add_noise
            num_coherent_signal = 0
        else:
            add_noise = np.random.randn(M, N) + 1j * np.random.randn(M, N)
            SNR = float('-inf')
            max_snr_index = -1
            num_coherent_signal = 0
            for i, c in enumerate(coherency_list):
                if c == 1:
                    num_coherent_signal += 1
                    if snr_list[i] > SNR:
                        SNR = snr_list[i]
                        max_snr_index = i
            main_coherent_signal = (10 ** (SNR / 20)) * (np.random.randn(1, N) + 1j * np.random.randn(1, N))
            coherent_coefs_list = []
            array_signal = 0
            array_signal_list = []
            for i, doa in enumerate(doa_list):
                if i == max_snr_index:
                    array_signal_i = self.generate_single_signal_array(doa, SNR, M, N, d,
                                                                       wavelength,
                                                                       main_coherent_signal)
                    array_signal_list.append(array_signal_i)
                    array_signal += array_signal_i
                    coherent_coefs_list.append((1, 0))
                elif coherency_list[i] == 1:
                    snr_diff = SNR - snr_list[i]
                    a = 10 ** (snr_diff / 20)
                    b = 2 * np.pi * np.random.rand()
                    signal = (1 / a) * np.exp(1j * b) * main_coherent_signal
                    array_signal_i = self.generate_single_signal_array(doa, snr_list[i], M, N, d,
                                                                       wavelength,
                                                                       signal)
                    array_signal_list.append(array_signal_i)
                    array_signal += array_signal_i
                    coherent_coefs_list.append((1 / a, b))
                else:
                    array_signal_i = self.generate_single_signal_array(doa, snr_list[i], M, N, d,
                                                                       wavelength)
                    array_signal_list.append(array_signal_i)
                    array_signal += array_signal_i
                    coherent_coefs_list.append((0, 0))
            array_output_nf = array_signal + 0 * add_noise  # noise-free output
            array_output = array_signal + 1 * add_noise
        samples_list = [{"doa_list": doa_list, "array_output": array_output, "array_output_nf": array_output_nf,
                         "array_signal_list": array_signal_list, "snr_list": snr_list, "snr_diff": None,
                         "coherent_coefs_list": coherent_coefs_list}]
        dataset = {"samples_list": samples_list, "M": M, "N": N, "K": K, "d": d, "wavelength": wavelength,
                   "doa_min": doa_min, "doa_max": doa_max, "num_rep": None, "step": None,
                   "max_num_samples": None, "coherent_signals_max_snr_diff": None,
                   "doa_delta_list": None, "snr_diff_list": None,
                   "num_coherent_signal": num_coherent_signal, "imperfections": []}
        return dataset

    def add_imperfections_to_samples(self, dataset, gain_inconsistence=0, phase_inconsistence=0,
                                     sensor_position_error=0, mutual_coupling=0,
                                     gain_inconsistence_array=None, phase_inconsistence_array=None,
                                     sensor_position_error_array=None, mutual_coupling_array=None):
        M = dataset["M"]
        assert gain_inconsistence_array is None or len(gain_inconsistence_array) == M - 1
        assert phase_inconsistence_array is None or len(phase_inconsistence_array) == M - 1
        assert sensor_position_error_array is None or len(sensor_position_error_array) == M - 1
        assert mutual_coupling_array is None or len(mutual_coupling_array) == M - 1
        samples_list = dataset["samples_list"]
        imperfection_matrix = np.eye(M)
        if phase_inconsistence != 0:
            if phase_inconsistence_array is None:
                phase_inconsistence_array = [0] + [-np.pi / 6] * math.ceil((M - 1) / 2) + [np.pi / 6] * math.floor(
                    (M - 1) / 2)
                phase_inconsistence_array = np.array(phase_inconsistence_array)
            else:
                phase_inconsistence_array = np.concatenate((np.array([0]), phase_inconsistence_array))
            phase_inconsistence_array = phase_inconsistence * phase_inconsistence_array
            phase_inconsistence_matrix = np.diag(np.exp(1j * phase_inconsistence_array))
            imperfection_matrix = np.matmul(phase_inconsistence_matrix, imperfection_matrix)
        if gain_inconsistence != 0:
            if gain_inconsistence_array is None:
                gain_inconsistence_array = [0] + [0.2] * math.ceil((M - 1) / 2) + [-0.2] * math.floor((M - 1) / 2)
                gain_inconsistence_array = np.array(gain_inconsistence_array)
            else:
                gain_inconsistence_array = np.concatenate((np.array([0]), gain_inconsistence_array))
            gain_inconsistence_array = gain_inconsistence * gain_inconsistence_array
            gain_inconsistence_matrix = np.eye(M) + np.diag(gain_inconsistence_array)
            imperfection_matrix = np.matmul(gain_inconsistence_matrix, imperfection_matrix)
        if mutual_coupling != 0:
            if mutual_coupling_array is None:
                gama = 0.3 * np.exp(1j * np.pi / 3)
                mutual_coupling_array = [0] + [gama ** i for i in range(1, M)]
                mutual_coupling_array = np.array(mutual_coupling_array)
            else:
                mutual_coupling_array = np.concatenate((np.array([0]), mutual_coupling_array))
            mutual_coupling_array = mutual_coupling * mutual_coupling_array
            mutual_coupling_matrix = np.eye(M) + linalg.toeplitz(mutual_coupling_array)
            imperfection_matrix = np.matmul(mutual_coupling_matrix, imperfection_matrix)
        if sensor_position_error != 0:
            if sensor_position_error_array is None:
                sensor_position_error_array = [0] + [-0.2] * math.ceil((M - 1) / 2) + [0.2] * math.floor((M - 1) / 2)
                sensor_position_error_array = np.array(sensor_position_error_array)
            else:
                sensor_position_error_array = np.concatenate((np.array([0]), sensor_position_error_array))
            sensor_position_error_array = sensor_position_error * sensor_position_error_array
            d = dataset["d"]
            wavelength = dataset["wavelength"]
            for index, sample in enumerate(samples_list):
                array_signal_list = sample["array_signal_list"]
                doa_list = sample["doa_list"]
                add_noise = sample["array_output"] - sample["array_output_nf"]
                array_signal = self.add_sensor_position_error(array_signal_list, doa_list, M, d, wavelength,
                                                              sensor_position_error_array)
                array_signal = np.matmul(imperfection_matrix, array_signal)
                samples_list[index]["array_output_nf"] = array_signal + 0 * add_noise
                samples_list[index]["array_output"] = array_signal + 1 * add_noise
        else:
            for index, sample in enumerate(samples_list):
                add_noise = sample["array_output"] - sample["array_output_nf"]
                array_signal = sample["array_output_nf"]
                array_signal = np.matmul(imperfection_matrix, array_signal)
                samples_list[index]["array_output_nf"] = array_signal + 0 * add_noise
                samples_list[index]["array_output"] = array_signal + 1 * add_noise
        dataset["samples_list"] = samples_list
        imperfection_dict = {}
        if gain_inconsistence != 0:
            imperfection_dict.update({"gain_inconsistence": {"gain_inconsistence": gain_inconsistence,
                                                             "gain_inconsistence_array": gain_inconsistence_array.tolist()}})
        if phase_inconsistence != 0:
            imperfection_dict.update({"phase_inconsistence": {"phase_inconsistence": phase_inconsistence,
                                                              "phase_inconsistence_array": phase_inconsistence_array.tolist()}})
        if sensor_position_error != 0:
            imperfection_dict.update({"sensor_position_error": {"sensor_position_error": sensor_position_error,
                                                                "sensor_position_error_array": sensor_position_error_array.tolist()}})
        if mutual_coupling != 0:
            imperfection_dict.update({"mutual_coupling": {"mutual_coupling": mutual_coupling,
                                                          "mutual_coupling_array": mutual_coupling_array.tolist()}})
        dataset["imperfections"] = imperfection_dict

    def add_sensor_position_error(self, array_signal_list, doa_list, M, d, wavelength, sensor_position_error_array):
        array_signal = 0
        for i, doa in enumerate(doa_list):
            array_signal_i = self.generate_single_signal_array_sp_error(array_signal_list[i][0:1, :], doa, M, d,
                                                                        wavelength, sensor_position_error_array)
            array_signal += array_signal_i
        return array_signal

    @staticmethod
    def generate_single_signal_array_sp_error(array_signal_original, doa, M, d, wavelength,
                                              sensor_position_error_array):
        phase_shift_unit = 2 * np.pi * d / wavelength * np.sin((doa / 180) * np.pi)
        antennas_pos_array = np.array(range(M)) + sensor_position_error_array
        a_ = np.cos(antennas_pos_array * phase_shift_unit) + 1j * np.sin(antennas_pos_array * phase_shift_unit)
        a = np.expand_dims(a_, axis=-1)
        array_signal = np.matmul(a, array_signal_original)
        return array_signal


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="..\datasets\dataset6_sections\dataset6")
    parser.add_argument("-train_ratio", type=float, default=0.8)
    parser.add_argument("-test_ratio", type=float, default=0.1)
    parser.add_argument("-M", type=int, default=10)
    parser.add_argument("-N", type=int, default=400)
    parser.add_argument("-K", type=int, default=2)
    parser.add_argument("--SNR", type=int, default=10)
    parser.add_argument("--wavelength", type=float, default=2)
    parser.add_argument("--doa_min", type=int, default=-60)
    parser.add_argument("--doa_max", type=int, default=60)
    parser.add_argument("--num_rep", type=int, default=6)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--num_coherent_signal", type=int, default=0)
    parser.add_argument("--coherent_signals_max_snr_diff", type=float, default=4)
    parser.add_argument("--snr_diff_list", default=None)
    args = parser.parse_args()

    # doa_delta_list = list(range(2, 41, 2))
    doa_delta_list = None
    d = args.wavelength / 2
    dataset_path = f'{args.dataset_path}_ULA_M_{args.M}_N_{args.N}_K_{args.K}_SNR_{args.SNR}_num_rep_{args.num_rep}_step_{args.step}_num_coherent_signal_{args.num_coherent_signal}'
    GD_instance = GenerateDatasetULA()
    GD_instance.generate_dataset_const_n_k_snr(output_path=dataset_path, M=args.M, N=args.N, K=args.K, SNR=args.SNR,
                                               d=d, wavelength=args.wavelength,
                                               doa_min=args.doa_min, doa_max=args.doa_max, num_rep=args.num_rep,
                                               step=args.step,
                                               coherent_signals_max_snr_diff=args.coherent_signals_max_snr_diff,
                                               doa_delta_list=doa_delta_list, snr_diff_list=args.snr_diff_list,
                                               num_coherent_signal=args.num_coherent_signal,
                                               display=True, train_ratio=args.train_ratio, test_ratio=args.test_ratio)

    # GD_instance = GenerateDatasetULA()
    # d = args.wavelength / 2
    # imperfections_list = [(1, 0, 0, 0, 0), (0, 1, 0, 0, 0), (0, 0, 1, 0, 0), (0, 0, 0, 1, 0), (0, 0, 0, 0, 0),
    #                       (0.5, 0.5, 0.5, 0.5, 0), (1, 1, 1, 1, 0), (0, 0, 0, 0, 2)]
    # K_list = [1]
    # snr_list = [-5, -2, 0, 2, 5, 10, 15, 20]
    # N_list = [1, 5, 10, 50, 100, 400]
    # # snr_diff_list = [0, 5]
    # snr_diff_list = [0]
    # # num_rep_dict = {2: 0.013889, 3: 0.000093, 4: 0.000093}
    # num_rep = 1
    # for gi, pi, sp, mc, co in tqdm(imperfections_list):
    #     if co == 2:
    #         continue
    #     for K in K_list:
    #         # num_rep = num_rep_dict[K]
    #         for SNR in snr_list:
    #             for N in N_list:
    #                 dataset_path = f'{args.dataset_path}_ULA_M_{args.M}_N_{N}_K_{K}_SNR_{SNR}_num_rep_{num_rep}_step_{args.step}_num_coherent_signal_{co}_gi_{gi}_pi_{pi}_sp_{sp}_mc_{mc}'
    #                 GD_instance.generate_dataset_const_n_k_snr(output_path=dataset_path, M=args.M, N=N, K=K, SNR=SNR,
    #                                                            d=d, wavelength=args.wavelength,
    #                                                            doa_min=args.doa_min, doa_max=args.doa_max,
    #                                                            num_rep=num_rep,
    #                                                            step=args.step,
    #                                                            coherent_signals_max_snr_diff=args.coherent_signals_max_snr_diff,
    #                                                            doa_delta_list=None, snr_diff_list=snr_diff_list,
    #                                                            num_coherent_signal=co, gain_inconsistence=gi,
    #                                                            phase_inconsistence=pi,
    #                                                            sensor_position_error=sp, mutual_coupling=mc,
    #                                                            display=True, train_ratio=args.train_ratio,
    #                                                            test_ratio=args.test_ratio)
