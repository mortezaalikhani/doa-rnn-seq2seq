"""
This module contains a class that is for evaluation and comparison various doa estimation methods.

"""

import json
import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import torch

from .generate_dataset import GenerateDatasetULA
from .utils import cal_mae, cal_rmse


class ComparisonAndEvaluation():
    """
    This class is for evaluation and comparison various doa estimation methods.

    """

    def __init__(self):
        self.GD_instance = GenerateDatasetULA()

    def evaluate_based_on_snr(self, interface_func, dataset_path=None, save_dataset=False, dataset_save_path=None,
                              save_results=False, results_save_path=None, doa_delta_min=None, eval_metric="mae",
                              snr_min=-5, snr_max=20, snr_step=1, num_samples_for_each_snr=1000, M=10, N=400, K=2, d=1,
                              wavelength=2, doa_min=-60, doa_max=60, num_coherent_signal=0):
        if eval_metric == "mae":
            eval_func = cal_mae
        elif eval_metric == "rmse":
            eval_func = cal_rmse
        else:
            print("The value for the eval_metric argument is invalid")
            exit()
        if dataset_path is not None:
            with open(dataset_path, "rb") as f:
                dataset = pickle.load(f)
        else:
            dataset = []
            coherency_list = [1] * num_coherent_signal + [0] * (K - num_coherent_signal)
            for snr in range(snr_min, snr_max + 1, snr_step):
                snr_list = [snr] * K
                for i in range(num_samples_for_each_snr):
                    random.shuffle(coherency_list)
                    if doa_delta_min is not None and (K in [2, 3]):
                        first_doa = np.random.choice(range(doa_min, doa_max - doa_delta_min))
                        second_doa = first_doa + doa_delta_min
                        doa_list = [first_doa, second_doa]
                        if K == 3:
                            third_doa_range = list(range(doa_min, first_doa - doa_delta_min + 1)) + list(
                                range(second_doa + doa_delta_min, doa_max))
                            if len(third_doa_range) == 0:
                                print(
                                    f"The doa_delta_min : {doa_delta_min} is not possible with three targets in doa range : [{doa_min}, {doa_max})")
                                exit()
                            else:
                                third_doa = np.random.choice(third_doa_range)
                                if third_doa < first_doa:
                                    doa_list = [third_doa] + doa_list
                                else:
                                    doa_list = doa_list + [third_doa]
                    else:
                        doa_list = np.sort(np.random.choice(range(doa_min, doa_max), K, replace=False), axis=-1,
                                           kind=None, order=None)
                    sample = self.GD_instance.generate_one_sample(M=M, N=N, doa_list=doa_list,
                                                                  snr_list=snr_list, d=d,
                                                                  wavelength=wavelength, doa_min=doa_min,
                                                                  doa_max=doa_max,
                                                                  coherency_list=coherency_list)
                    sample = sample['samples_list'][0]
                    doa_list = sample["doa_list"]
                    target_sequence = [doa - doa_min for doa in doa_list]
                    target_vector = [0] * (doa_max - doa_min)
                    for t in target_sequence:
                        target_vector[t] = 1
                    sample_info = {"doa_list": sample["doa_list"], "snr_list": sample["snr_list"],
                                   "snr_diff": sample["snr_diff"],
                                   "coherent_coefs_list": sample["coherent_coefs_list"], "M": M, "N": N, "K": K, "d": d,
                                   "wavelength": wavelength,
                                   "num_coherent_signal": num_coherent_signal, "imperfections": []}
                    dataset.append({"array_output": sample["array_output"], "target_sequence": target_sequence,
                                    "target_vector": target_vector, "sample_info": sample_info})
        if save_dataset:
            os.makedirs(os.path.dirname(dataset_save_path), exist_ok=True)
            with open(dataset_save_path, "wb") as f:
                pickle.dump(dataset, f)
        interface_output = interface_func(dataset)
        doa_errors_based_on_snr = {}
        for out in interface_output["output_list"]:
            if out["sample_info"]["snr_list"][0] in doa_errors_based_on_snr:
                doa_errors_based_on_snr[out["sample_info"]["snr_list"][0]] = torch.hstack(
                    (doa_errors_based_on_snr[out["sample_info"]["snr_list"][0]], out["doa_errors"]))
            else:
                doa_errors_based_on_snr[out["sample_info"]["snr_list"][0]] = out["doa_errors"]
        metric_based_on_snr = {}
        for snr in doa_errors_based_on_snr:
            metric_based_on_snr[snr] = eval_func(doa_errors_based_on_snr[snr])
        if save_results:
            os.makedirs(os.path.dirname(results_save_path), exist_ok=True)
            with open(results_save_path, "w") as f:
                json.dump(metric_based_on_snr, f)
        return metric_based_on_snr

    def compare_based_on_snr(self, algorithms_outputs, metric="mae"):
        self.compare_algorithms(algorithms_outputs, metric, "snr")

    def evaluate_with_array_imperfections(self, interface_func, dataset_path=None, save_dataset=False,
                                          dataset_save_path=None,
                                          save_results=False, results_save_path=None, doa_delta_min=None,
                                          eval_metric="mae",
                                          num_samples_for_each_rho=500, M=10, N=400, K=2, SNR=10, d=1,
                                          wavelength=2, doa_min=-60, doa_max=60, num_coherent_signal=0,
                                          gain_inconsistence_en=0, phase_inconsistence_en=0,
                                          sensor_position_error_en=0, mutual_coupling_en=0,
                                          gain_inconsistence_array=None, phase_inconsistence_array=None,
                                          sensor_position_error_array=None, mutual_coupling_array=None, rho_min=0,
                                          rho_max=1, num_rho_points=11):
        if eval_metric == "mae":
            eval_func = cal_mae
        elif eval_metric == "rmse":
            eval_func = cal_rmse
        else:
            print("The value for the eval_metric argument is invalid")
            exit()
        if dataset_path is not None:
            with open(dataset_path, "rb") as f:
                dataset = pickle.load(f)
        else:
            dataset = []
            coherency_list = [1] * num_coherent_signal + [0] * (K - num_coherent_signal)
            snr_list = [SNR] * K
            for rho in np.linspace(rho_min, rho_max, num_rho_points):
                for i in range(num_samples_for_each_rho):
                    random.shuffle(coherency_list)
                    if doa_delta_min is not None and (K in [2, 3]):
                        first_doa = np.random.choice(range(doa_min, doa_max - doa_delta_min))
                        second_doa = first_doa + doa_delta_min
                        doa_list = [first_doa, second_doa]
                        if K == 3:
                            third_doa_range = list(range(doa_min, first_doa - doa_delta_min + 1)) + list(
                                range(second_doa + doa_delta_min, doa_max))
                            if len(third_doa_range) == 0:
                                print(
                                    f"The doa_delta_min : {doa_delta_min} is not possible with three targets in doa range : [{doa_min}, {doa_max})")
                                exit()
                            else:
                                third_doa = np.random.choice(third_doa_range)
                                if third_doa < first_doa:
                                    doa_list = [third_doa] + doa_list
                                else:
                                    doa_list = doa_list + [third_doa]
                    else:
                        doa_list = np.sort(np.random.choice(range(doa_min, doa_max), K, replace=False), axis=-1,
                                           kind=None, order=None)
                    sample = self.GD_instance.generate_one_sample(M=M, N=N, doa_list=doa_list,
                                                                  snr_list=snr_list, d=d,
                                                                  wavelength=wavelength, doa_min=doa_min,
                                                                  doa_max=doa_max,
                                                                  coherency_list=coherency_list)
                    self.GD_instance.add_imperfections_to_samples(sample, gain_inconsistence_en * rho,
                                                                  phase_inconsistence_en * rho,
                                                                  sensor_position_error_en * rho,
                                                                  mutual_coupling_en * rho, gain_inconsistence_array,
                                                                  phase_inconsistence_array,
                                                                  sensor_position_error_array, mutual_coupling_array)
                    imperfections = sample["imperfections"]
                    sample = sample['samples_list'][0]
                    doa_list = sample["doa_list"]
                    target_sequence = [doa - doa_min for doa in doa_list]
                    target_vector = [0] * (doa_max - doa_min)
                    for t in target_sequence:
                        target_vector[t] = 1
                    sample_info = {"doa_list": sample["doa_list"], "snr_list": sample["snr_list"],
                                   "snr_diff": sample["snr_diff"],
                                   "coherent_coefs_list": sample["coherent_coefs_list"], "M": M, "N": N, "K": K, "d": d,
                                   "wavelength": wavelength,
                                   "num_coherent_signal": num_coherent_signal, "imperfections": imperfections}
                    dataset.append({"array_output": sample["array_output"], "target_sequence": target_sequence,
                                    "target_vector": target_vector, "sample_info": sample_info})
        if save_dataset:
            os.makedirs(os.path.dirname(dataset_save_path), exist_ok=True)
            with open(dataset_save_path, "wb") as f:
                pickle.dump(dataset, f)
        interface_output = interface_func(dataset)
        doa_errors_based_on_rho = {}
        for out in interface_output["output_list"]:
            imperfections = out["sample_info"]["imperfections"]
            if not isinstance(imperfections, dict):
                rho = 0
            elif "gain_inconsistence" in imperfections:
                rho = imperfections["gain_inconsistence"]["gain_inconsistence"]
            elif "phase_inconsistence" in imperfections:
                rho = imperfections["phase_inconsistence"]["phase_inconsistence"]
            elif "sensor_position_error" in imperfections:
                rho = imperfections["sensor_position_error"]["sensor_position_error"]
            elif "mutual_coupling" in imperfections:
                rho = imperfections["mutual_coupling"]["mutual_coupling"]
            else:
                rho = 0
            if rho in doa_errors_based_on_rho:
                doa_errors_based_on_rho[rho] = torch.hstack(
                    (doa_errors_based_on_rho[rho], out["doa_errors"]))
            else:
                doa_errors_based_on_rho[rho] = out["doa_errors"]
        metric_based_on_rho = {}
        for rho in doa_errors_based_on_rho:
            metric_based_on_rho[rho] = eval_func(doa_errors_based_on_rho[rho])
        if save_results:
            os.makedirs(os.path.dirname(results_save_path), exist_ok=True)
            with open(results_save_path, "w") as f:
                json.dump(metric_based_on_rho, f)
        return metric_based_on_rho

    def compare_based_on_rho(self, algorithms_outputs, metric="mae"):
        self.compare_algorithms(algorithms_outputs, metric, r"$\rho$")

    def evaluate_based_on_N(self, interface_func, dataset_path=None, save_dataset=False, dataset_save_path=None,
                            save_results=False, results_save_path=None, doa_delta_min=None, eval_metric="mae",
                            N_values=[1, 5, 10, 50, 100, 400], num_samples_for_each_N=1000, M=10, SNR=10, K=2, d=1,
                            wavelength=2, doa_min=-60, doa_max=60, num_coherent_signal=0):
        if eval_metric == "mae":
            eval_func = cal_mae
        elif eval_metric == "rmse":
            eval_func = cal_rmse
        else:
            print("The value for the eval_metric argument is invalid")
            exit()
        if dataset_path is not None:
            with open(dataset_path, "rb") as f:
                dataset = pickle.load(f)
        else:
            dataset = []
            coherency_list = [1] * num_coherent_signal + [0] * (K - num_coherent_signal)
            snr_list = [SNR] * K
            for N in N_values:
                for i in range(num_samples_for_each_N):
                    random.shuffle(coherency_list)
                    if doa_delta_min is not None and (K in [2, 3]):
                        first_doa = np.random.choice(range(doa_min, doa_max - doa_delta_min))
                        second_doa = first_doa + doa_delta_min
                        doa_list = [first_doa, second_doa]
                        if K == 3:
                            third_doa_range = list(range(doa_min, first_doa - doa_delta_min + 1)) + list(
                                range(second_doa + doa_delta_min, doa_max))
                            if len(third_doa_range) == 0:
                                print(
                                    f"The doa_delta_min : {doa_delta_min} is not possible with three targets in doa range : [{doa_min}, {doa_max})")
                                exit()
                            else:
                                third_doa = np.random.choice(third_doa_range)
                                if third_doa < first_doa:
                                    doa_list = [third_doa] + doa_list
                                else:
                                    doa_list = doa_list + [third_doa]
                    else:
                        doa_list = np.sort(np.random.choice(range(doa_min, doa_max), K, replace=False), axis=-1,
                                           kind=None, order=None)
                    sample = self.GD_instance.generate_one_sample(M=M, N=N, doa_list=doa_list,
                                                                  snr_list=snr_list, d=d,
                                                                  wavelength=wavelength, doa_min=doa_min,
                                                                  doa_max=doa_max,
                                                                  coherency_list=coherency_list)
                    sample = sample['samples_list'][0]
                    doa_list = sample["doa_list"]
                    target_sequence = [doa - doa_min for doa in doa_list]
                    target_vector = [0] * (doa_max - doa_min)
                    for t in target_sequence:
                        target_vector[t] = 1
                    sample_info = {"doa_list": sample["doa_list"], "snr_list": sample["snr_list"],
                                   "snr_diff": sample["snr_diff"],
                                   "coherent_coefs_list": sample["coherent_coefs_list"], "M": M, "N": N, "K": K, "d": d,
                                   "wavelength": wavelength,
                                   "num_coherent_signal": num_coherent_signal, "imperfections": []}
                    dataset.append({"array_output": sample["array_output"], "target_sequence": target_sequence,
                                    "target_vector": target_vector, "sample_info": sample_info})
        if save_dataset:
            os.makedirs(os.path.dirname(dataset_save_path), exist_ok=True)
            with open(dataset_save_path, "wb") as f:
                pickle.dump(dataset, f)
        interface_output = interface_func(dataset)
        doa_errors_based_on_N = {}
        for out in interface_output["output_list"]:
            if out["sample_info"]["N"] in doa_errors_based_on_N:
                doa_errors_based_on_N[out["sample_info"]["N"]] = torch.hstack(
                    (doa_errors_based_on_N[out["sample_info"]["N"]], out["doa_errors"]))
            else:
                doa_errors_based_on_N[out["sample_info"]["N"]] = out["doa_errors"]
        metric_based_on_N = {}
        for N in doa_errors_based_on_N:
            metric_based_on_N[N] = eval_func(doa_errors_based_on_N[N])
        if save_results:
            os.makedirs(os.path.dirname(results_save_path), exist_ok=True)
            with open(results_save_path, "w") as f:
                json.dump(metric_based_on_N, f)
        return metric_based_on_N

    def compare_based_on_N(self, algorithms_outputs, metric="mae"):
        self.compare_algorithms(algorithms_outputs, metric, "N")

    @staticmethod
    def compare_algorithms(algorithms_outputs, metric="mae", independent_variable="snr"):
        plt.figure(f"{metric} based on {independent_variable}")
        for index, (algorithm, output) in enumerate(algorithms_outputs):
            sorted_output = sorted(output.items(), key=lambda x: float(x[0]))
            x_list = [out[0][0:min(len(out[0]), 3)] for out in sorted_output]
            y_list = [out[1] for out in sorted_output]
            if index == 0:
                x_list_base = x_list
            else:
                if x_list != x_list_base:
                    print(f"The {independent_variable} lists of different algorithms are not equal")
                    exit()
            plt.plot(x_list_base, y_list, label=algorithm)
        plt.xlabel(independent_variable)
        plt.ylabel(metric)
        plt.legend(loc='upper right')
        plt.show()

    def evaluate_speed(self, interface_func, dataset_path=None, save_dataset=False, dataset_save_path=None,
                       save_results=False, results_save_path=None, doa_delta_min=None,
                       N_values=[1, 5, 10, 50, 100, 400], num_samples_for_each_N=1000, M=10, SNR=10, K=2, d=1,
                       wavelength=2, doa_min=-60, doa_max=60, num_coherent_signal=0):
        if dataset_path is not None:
            with open(dataset_path, "rb") as f:
                dataset = pickle.load(f)
        else:
            dataset = dict()
            coherency_list = [1] * num_coherent_signal + [0] * (K - num_coherent_signal)
            snr_list = [SNR] * K
            for N in N_values:
                dataset[N] = []
                for i in range(num_samples_for_each_N):
                    random.shuffle(coherency_list)
                    if doa_delta_min is not None and (K in [2, 3]):
                        first_doa = np.random.choice(range(doa_min, doa_max - doa_delta_min))
                        second_doa = first_doa + doa_delta_min
                        doa_list = [first_doa, second_doa]
                        if K == 3:
                            third_doa_range = list(range(doa_min, first_doa - doa_delta_min + 1)) + list(
                                range(second_doa + doa_delta_min, doa_max))
                            if len(third_doa_range) == 0:
                                print(
                                    f"The doa_delta_min : {doa_delta_min} is not possible with three targets in doa range : [{doa_min}, {doa_max})")
                                exit()
                            else:
                                third_doa = np.random.choice(third_doa_range)
                                if third_doa < first_doa:
                                    doa_list = [third_doa] + doa_list
                                else:
                                    doa_list = doa_list + [third_doa]
                    else:
                        doa_list = np.sort(np.random.choice(range(doa_min, doa_max), K, replace=False), axis=-1,
                                           kind=None, order=None)
                    sample = self.GD_instance.generate_one_sample(M=M, N=N, doa_list=doa_list,
                                                                  snr_list=snr_list, d=d,
                                                                  wavelength=wavelength, doa_min=doa_min,
                                                                  doa_max=doa_max,
                                                                  coherency_list=coherency_list)
                    sample = sample['samples_list'][0]
                    doa_list = sample["doa_list"]
                    target_sequence = [doa - doa_min for doa in doa_list]
                    target_vector = [0] * (doa_max - doa_min)
                    for t in target_sequence:
                        target_vector[t] = 1
                    sample_info = {"doa_list": sample["doa_list"], "snr_list": sample["snr_list"],
                                   "snr_diff": sample["snr_diff"],
                                   "coherent_coefs_list": sample["coherent_coefs_list"], "M": M, "N": N, "K": K, "d": d,
                                   "wavelength": wavelength,
                                   "num_coherent_signal": num_coherent_signal, "imperfections": []}
                    dataset[N].append({"array_output": sample["array_output"], "target_sequence": target_sequence,
                                       "target_vector": target_vector, "sample_info": sample_info})
        if save_dataset:
            os.makedirs(os.path.dirname(dataset_save_path), exist_ok=True)
            with open(dataset_save_path, "wb") as f:
                pickle.dump(dataset, f)
        out_dict = dict()
        for N in dataset:
            interface_output = interface_func(dataset[N])
            out_dict[N] = interface_output["total_time"]
        if save_results:
            os.makedirs(os.path.dirname(results_save_path), exist_ok=True)
            with open(results_save_path, "w") as f:
                json.dump(out_dict, f)
        return out_dict

    def evaluate_on_dataset(self, interface_func, dataset_path, save_results=False, results_save_path=None,
                            mae_metric=True, rmse_metric=True):
        with open(dataset_path, "rb") as f:
            dataset = pickle.load(f)
        dataset = [z for i, z in enumerate(dataset) if i % 5 == 0]
        interface_output = interface_func(dataset)
        doa_errors = [output["doa_errors"] for output in interface_output["output_list"]]
        doa_errors_tensor = torch.hstack(doa_errors)
        metrics = dict()
        if mae_metric:
            mae = cal_mae(doa_errors_tensor)
            metrics["mae"] = mae
        if rmse_metric:
            rmse = cal_rmse(doa_errors_tensor)
            metrics["rmse"] = rmse
        if save_results:
            os.makedirs(os.path.dirname(results_save_path), exist_ok=True)
            with open(results_save_path, "w") as f:
                json.dump(metrics, f)
        return metrics
