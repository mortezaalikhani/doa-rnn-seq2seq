"""
This module contains classes to predict and inference for one sample in the doa estimation problem.

"""

import argparse
import json
import math
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from net.Attention import Attention
from net.BIGRUClassifier import BIGRUClassifier
from net.BILSTMClassifier import BILSTMClassifier
from net.Decoder import Decoder
from net.Encoder import Encoder
from net.GRUClassifier import GRUClassifier
from net.LSTMClassifier import LSTMClassifier
from net.Seq2Seq import Seq2Seq
from utils import generate_dataset
from utils import utils


class PredictClassifier():
    """
    This Class is for predicting and inferencing for one sample in the doa estimation problem by RNN classifiers.

    """

    def predict(self, model, data, target, criterion, device, doa_est_func, doa_min=-60, probs_fig_save_address=None,
                num_targets_known=True, doa_probs_threshold=1e-6):
        model.eval()

        data = data.to(device)
        target = target.to(device)

        output = model(data)

        loss = criterion(output, target)

        sample_loss = loss.item()

        output = output.to('cpu')
        target = target.to('cpu')

        probs_list = torch.reshape(output, [-1]).tolist()

        if num_targets_known:
            output, estimated_doas_list = doa_est_func(output, target)
        else:
            output, estimated_doas_list = doa_est_func(output, doa_probs_threshold)

        doa_errors = utils.cal_doa_errors(estimated_doas_list, target)
        mae = utils.cal_mae(doa_errors)
        rmse = utils.cal_rmse(doa_errors)

        estimated_doas_list = estimated_doas_list[0]
        estimated_doas_list = [int(doa) + doa_min for doa, prob in estimated_doas_list]

        utils.plot_probs(probs_list, doa_min, probs_fig_save_address=probs_fig_save_address, title="DOA probs",
                         x_label="doa", y_label="prob")

        return sample_loss, mae, rmse, estimated_doas_list

    def predict_model(self, model, data, target, criterion, device, doa_est_func, doa_min=-60, out_dir=None,
                      num_targets_known=True, doa_probs_threshold=1e-6):
        res_save_address = None
        probs_fig_save_address = None
        if out_dir is not None:
            res_save_address = os.path.join(out_dir, "results.json")
            probs_fig_save_address = os.path.join(out_dir, "probs.png")

        sample_loss, mae, rmse, estimated_doas_list = self.predict(model, data, target, criterion, device,
                                                                   doa_est_func, doa_min=doa_min,
                                                                   probs_fig_save_address=probs_fig_save_address,
                                                                   num_targets_known=num_targets_known,
                                                                   doa_probs_threshold=doa_probs_threshold)

        print(f'| Sample Loss: {sample_loss:.3f} | Sample PPL: {math.exp(sample_loss):7.3f} |')
        print(f'\t MAE: {mae:.3f}')
        print(f'\t RMSE: {rmse:.3f}')
        print(f'\t estimated doas list: {estimated_doas_list}')

        if res_save_address is not None:
            res_dict = {"sample_loss": sample_loss, "mae": mae, "rmse": rmse,
                        "estimated_doas_list": estimated_doas_list}
            with open(res_save_address, "w") as f:
                json.dump(res_dict, f)


class PredictSeq2SeqNetwork():
    """
    This Class is for predicting and inferencing for one sample in the doa estimation problem by Seq2Seq networks.

    """

    def predict(self, model, data, target, criterion, device, doa_est_func, doa_min=-60, probs_fig_save_address=None):
        model.eval()

        data = data.to(device)
        target = target.to(device)

        output = model(data, target, 0)

        output_dim = output.shape[-1]

        output_flatten = output[1:].view(-1, output_dim)
        trg_flatten = torch.reshape(target[1:], (-1,))

        loss = criterion(output_flatten, trg_flatten)

        sample_loss = loss.item()

        output = output.to('cpu')
        target = target.to('cpu')

        output, estimated_doas_list, target, prob = doa_est_func(output, target)

        probs_list = torch.reshape(prob, [-1]).tolist()

        doa_errors = utils.cal_doa_errors(estimated_doas_list, target)
        mae = utils.cal_mae(doa_errors)
        rmse = utils.cal_rmse(doa_errors)

        estimated_doas_list = estimated_doas_list[0]
        estimated_doas_list = [int(doa) + doa_min for doa, prob in estimated_doas_list]

        utils.plot_probs(probs_list, doa_min, probs_fig_save_address=probs_fig_save_address, title="DOA probs",
                         x_label="doa", y_label="prob")

        return sample_loss, mae, rmse, estimated_doas_list

    def predict_model(self, model, data, target, criterion, device, doa_est_func, doa_min=-60, out_dir=None):
        res_save_address = None
        probs_fig_save_address = None
        if out_dir is not None:
            res_save_address = os.path.join(out_dir, "results.json")
            probs_fig_save_address = os.path.join(out_dir, "probs.png")

        sample_loss, mae, rmse, estimated_doas_list = self.evaluate(model, data, target, criterion, device,
                                                                    doa_est_func, doa_min=doa_min,
                                                                    probs_fig_save_address=probs_fig_save_address)

        print(f'| Sample Loss: {sample_loss:.3f} | Sample PPL: {math.exp(sample_loss):7.3f} |')
        print(f'\t MAE: {mae:.3f}')
        print(f'\t RMSE: {rmse:.3f}')
        print(f'\t estimated doas list: {estimated_doas_list}')

        if res_save_address is not None:
            res_dict = {"sample_loss": sample_loss, "mae": mae, "rmse": rmse,
                        "estimated_doas_list": estimated_doas_list}
            with open(res_save_address, "w") as f:
                json.dump(res_dict, f)


def main(args):
    if args.out_dir is not None:
        os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device : {device}')

    model_params_address = os.path.join(args.model_dir, "params.csv")
    params_df = pd.read_csv(model_params_address)

    d = args.wavelength / 2

    GD_instance = generate_dataset.GenerateDatasetULA()
    sample = GD_instance.generate_one_sample(M=args.M, N=args.N, doa_list=args.doa_list, snr_list=args.snr_list, d=d,
                                             wavelength=args.wavelength, doa_min=args.doa_min, doa_max=args.doa_max,
                                             coherency_list=args.coherency_list)
    sample = sample['samples_list'][0]

    if args.dataset_type == 'SequenceOfSnapshotsFixedN' or args.dataset_type == 'SequenceOfSnapshotsVariableN':
        features_array = sample['array_output']
        features_real = np.real(features_array)
        features_imag = np.imag(features_array)
        features = torch.from_numpy(np.transpose(np.concatenate((features_real, features_imag), axis=0))).float()
        data = torch.unsqueeze(features, 1)

        doa_list = sample["doa_list"]
        target_sequence = [doa - args.doa_min for doa in doa_list]
        target_vector = [0] * (args.doa_max - args.doa_min)
        for i in target_sequence:
            target_vector[i] = 1
        target = torch.FloatTensor(target_vector)
        target = torch.unsqueeze(target, 0)
        kwargs = {'input_dim': int(params_df.loc[params_df['keys'] == 'input_dim', 'values']),
                  'dense_size': int(params_df.loc[params_df['keys'] == 'dense_size', 'values']),
                  'hid_dim': int(params_df.loc[params_df['keys'] == 'hidden_units', 'values']),
                  'num_layer': int(params_df.loc[params_df['keys'] == 'num_layers', 'values']),
                  'num_classes': int(params_df.loc[params_df['keys'] == 'num_classes', 'values']),
                  'dropout': float(params_df.loc[params_df['keys'] == 'dropout', 'values']),
                  'seed': int(params_df.loc[params_df['keys'] == 'seed', 'values'])}
    elif args.dataset_type == 'SequenceOfAntennas':
        features_array = sample['array_output']
        features_real = np.real(features_array)
        features_imag = np.imag(features_array)
        features = torch.from_numpy(np.concatenate((features_real, features_imag), axis=0)).float()
        data = torch.unsqueeze(features, 1)

        doa_list = sample["doa_list"]
        target_sequence = [doa - args.doa_min for doa in doa_list]
        target_vector = [0] * (args.doa_max - args.doa_min)
        for i in target_sequence:
            target_vector[i] = 1
        target = torch.FloatTensor(target_vector)
        target = torch.unsqueeze(target, 0)
        kwargs = {'input_dim': int(params_df.loc[params_df['keys'] == 'input_dim', 'values']),
                  'dense_size': int(params_df.loc[params_df['keys'] == 'dense_size', 'values']),
                  'hid_dim': int(params_df.loc[params_df['keys'] == 'hidden_units', 'values']),
                  'num_layer': int(params_df.loc[params_df['keys'] == 'num_layers', 'values']),
                  'num_classes': int(params_df.loc[params_df['keys'] == 'num_classes', 'values']),
                  'dropout': float(params_df.loc[params_df['keys'] == 'dropout', 'values']),
                  'seed': int(params_df.loc[params_df['keys'] == 'seed', 'values'])}
    if args.dataset_type == 'SequenceOfCovMatrixRows':
        features_array = sample['array_output']
        features_cov = np.cov(features_array)
        features_cov_real = np.real(features_cov)
        features_cov_imag = np.imag(features_cov)
        features = torch.from_numpy(np.concatenate((features_cov_real, features_cov_imag), axis=1)).float()
        data = torch.unsqueeze(features, 1)

        doa_list = sample["doa_list"]
        target_sequence = [doa - args.doa_min for doa in doa_list]
        target_vector = [0] * (args.doa_max - args.doa_min)
        for i in target_sequence:
            target_vector[i] = 1
        target = torch.FloatTensor(target_vector)
        target = torch.unsqueeze(target, 0)
        kwargs = {'input_dim': int(params_df.loc[params_df['keys'] == 'input_dim', 'values']),
                  'dense_size': int(params_df.loc[params_df['keys'] == 'dense_size', 'values']),
                  'hid_dim': int(params_df.loc[params_df['keys'] == 'hidden_units', 'values']),
                  'num_layer': int(params_df.loc[params_df['keys'] == 'num_layers', 'values']),
                  'num_classes': int(params_df.loc[params_df['keys'] == 'num_classes', 'values']),
                  'dropout': float(params_df.loc[params_df['keys'] == 'dropout', 'values']),
                  'seed': int(params_df.loc[params_df['keys'] == 'seed', 'values'])}
    elif args.dataset_type == 'SequenceOfSnapshotsFixedNForSeq2Seq' or args.dataset_type == 'SequenceOfSnapshotsVariableNForSeq2Seq':
        features_array = sample['array_output']
        features_real = np.real(features_array)
        features_imag = np.imag(features_array)
        features = torch.from_numpy(np.transpose(np.concatenate((features_real, features_imag), axis=0))).float()
        data = torch.unsqueeze(features, 1)

        doa_list = sample["doa_list"]
        target_sequence = [doa - args.doa_min for doa in doa_list]
        target = torch.LongTensor(target_sequence)
        target = target + 2
        target = torch.hstack((torch.LongTensor([0]), target,
                               torch.LongTensor(
                                   [1] * (int(params_df.loc[params_df['keys'] == 'trg_len', 'values']) - len(
                                       target) - 1))))
        target = torch.unsqueeze(target, 1)
        kwargs = {'input_dim': int(params_df.loc[params_df['keys'] == 'input_dim', 'values']),
                  'emb_dim': int(params_df.loc[params_df['keys'] == 'dense_size', 'values']),
                  'enc_hid_dim': int(params_df.loc[params_df['keys'] == 'hidden_units', 'values']),
                  'dec_hid_dim': int(params_df.loc[params_df['keys'] == 'hidden_units_dec', 'values']),
                  'dropout': float(params_df.loc[params_df['keys'] == 'dropout', 'values']),
                  'num_layers': int(params_df.loc[params_df['keys'] == 'num_layers', 'values']),
                  'seed': int(params_df.loc[params_df['keys'] == 'seed', 'values']),
                  'output_dim': int(params_df.loc[params_df['keys'] == 'output_dim', 'values'])}
        kwargs_encoder = {'input_dim': kwargs['input_dim'],
                          'emb_dim': kwargs['emb_dim'],
                          'enc_hid_dim': kwargs['enc_hid_dim'],
                          'dec_hid_dim': kwargs['dec_hid_dim'],
                          'dropout': kwargs['dropout'],
                          'num_layers': kwargs['num_layers'],
                          'seed': kwargs['seed']}
        kwargs_attention = {'enc_hid_dim': kwargs['enc_hid_dim'],
                            'dec_hid_dim': kwargs['dec_hid_dim'],
                            'seed': kwargs['seed']}
        kwargs_decoder = {'output_dim': kwargs['output_dim'],
                          'emb_dim': kwargs['emb_dim'],
                          'enc_hid_dim': kwargs['enc_hid_dim'],
                          'dec_hid_dim': kwargs['dec_hid_dim'],
                          'dropout': kwargs['dropout'],
                          'seed': kwargs['seed']}
    elif args.dataset_type == 'SequenceOfCovMatrixRowsForSeq2Seq':
        features_array = sample['array_output']
        features_cov = np.cov(features_array)
        features_cov_real = np.real(features_cov)
        features_cov_imag = np.imag(features_cov)
        features = torch.from_numpy(np.concatenate((features_cov_real, features_cov_imag), axis=1)).float()
        data = torch.unsqueeze(features, 1)

        doa_list = sample["doa_list"]
        target_sequence = [doa - args.doa_min for doa in doa_list]
        target = torch.LongTensor(target_sequence)
        target = target + 2
        target = torch.hstack((torch.LongTensor([0]), target,
                               torch.LongTensor(
                                   [1] * (int(params_df.loc[params_df['keys'] == 'trg_len', 'values']) - len(
                                       target) - 1))))
        target = torch.unsqueeze(target, 1)
        kwargs = {'input_dim': int(params_df.loc[params_df['keys'] == 'input_dim', 'values']),
                  'emb_dim': int(params_df.loc[params_df['keys'] == 'dense_size', 'values']),
                  'enc_hid_dim': int(params_df.loc[params_df['keys'] == 'hidden_units', 'values']),
                  'dec_hid_dim': int(params_df.loc[params_df['keys'] == 'hidden_units_dec', 'values']),
                  'dropout': float(params_df.loc[params_df['keys'] == 'dropout', 'values']),
                  'num_layers': int(params_df.loc[params_df['keys'] == 'num_layers', 'values']),
                  'seed': int(params_df.loc[params_df['keys'] == 'seed', 'values']),
                  'output_dim': int(params_df.loc[params_df['keys'] == 'output_dim', 'values'])}
        kwargs_encoder = {'input_dim': kwargs['input_dim'],
                          'emb_dim': kwargs['emb_dim'],
                          'enc_hid_dim': kwargs['enc_hid_dim'],
                          'dec_hid_dim': kwargs['dec_hid_dim'],
                          'dropout': kwargs['dropout'],
                          'num_layers': kwargs['num_layers'],
                          'seed': kwargs['seed']}
        kwargs_attention = {'enc_hid_dim': kwargs['enc_hid_dim'],
                            'dec_hid_dim': kwargs['dec_hid_dim'],
                            'seed': kwargs['seed']}
        kwargs_decoder = {'output_dim': kwargs['output_dim'],
                          'emb_dim': kwargs['emb_dim'],
                          'enc_hid_dim': kwargs['enc_hid_dim'],
                          'dec_hid_dim': kwargs['dec_hid_dim'],
                          'dropout': kwargs['dropout'],
                          'seed': kwargs['seed']}
    else:
        print("The value for the dataset_type argument is invalid")
        exit()

    model_type = list(params_df.loc[params_df['keys'] == 'model_type', 'values'])[0]
    if model_type == 'BILSTM':
        model = BILSTMClassifier(**kwargs)
    elif model_type == 'LSTM':
        model = LSTMClassifier(**kwargs)
    elif model_type == 'BIGRU':
        model = BIGRUClassifier(**kwargs)
    elif model_type == 'GRU':
        model = GRUClassifier(**kwargs)
    elif model_type == 'GRU':
        model = GRUClassifier(**kwargs)
    elif model_type == 'Seq2Seq':
        encoder = Encoder(**kwargs_encoder)
        attention = Attention(**kwargs_attention)
        decoder = Decoder(attention=attention, **kwargs_decoder)
        model = Seq2Seq(encoder, decoder, device)
    else:
        print("The value for the model_type argument is invalid")
        exit()

    model_save_address = os.path.join(args.model_dir, 'model.pt')

    model = model.to(device)

    model.load_state_dict(torch.load(model_save_address))

    if args.loss_type == 'BCE':
        criterion = nn.BCELoss()
    elif args.loss_type == 'FOCAL':
        pass
    elif args.loss_type == 'CrossEntropy':
        criterion = nn.CrossEntropyLoss()
    else:
        print("The value for the loss_type argument is invalid")
        exit()

    if args.estimation_of_doas == 'MaxK':
        doa_est_func = utils.estimate_doas_max_k
        num_targets_known = True
    elif args.estimation_of_doas == 'LocalMaxK':
        doa_est_func = utils.estimate_doas_local_max_k
        num_targets_known = True
    elif args.estimation_of_doas == 'Seq2Seq':
        doa_est_func = utils.estimate_doas_seq2seq
    elif args.estimation_of_doas == 'LocalMax':
        doa_est_func = utils.estimate_doas_local_max
        num_targets_known = False
    else:
        print("The value for the estimation_of_doas argument is invalid")
        exit()

    if model_type == 'Seq2Seq':
        ts_instance = PredictSeq2SeqNetwork()
        ts_instance.predict_model(model, data, target, criterion, device, doa_est_func, args.doa_min, args.out_dir)
    else:
        tc_instance = PredictClassifier()
        tc_instance.predict_model(model, data, target, criterion, device, doa_est_func, args.doa_min, args.out_dir,
                                  num_targets_known, args.doa_probs_threshold)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-M", type=int, default=10)
    parser.add_argument("-N", type=int, default=10)
    parser.add_argument("--doa_list", type=list, default=[-15, -10, 0])
    parser.add_argument("--snr_list", type=list, default=[10, 10, 10])
    parser.add_argument("--wavelength", type=float, default=2)
    parser.add_argument("--doa_min", type=int, default=-60)
    parser.add_argument("--doa_max", type=int, default=60)
    parser.add_argument("--coherency_list", default=None)
    parser.add_argument('--model_dir', type=str, help='Directory of model')
    parser.add_argument('--out_dir', default=None, help='Path for directory that results will be saved in it')
    parser.add_argument('--dataset_type', type=str,
                        choices=['SequenceOfSnapshotsFixedN', 'SequenceOfSnapshotsVariableN', 'SequenceOfAntennas',
                                 'SequenceOfCovMatrixRows', 'SequenceOfSnapshotsFixedNForSeq2Seq',
                                 'SequenceOfSnapshotsVariableNForSeq2Seq', 'SequenceOfCovMatrixRowsForSeq2Seq'],
                        default='SequenceOfSnapshotsFixedN', help='dataset type')
    parser.add_argument('--loss_type', type=str, choices=['BCE', 'FOCAL', 'CrossEntropy'], default='BCE',
                        help='loss function type')
    parser.add_argument('--estimation_of_doas', type=str, choices=['MaxK', 'LocalMaxK', 'Seq2Seq', 'LocalMax'],
                        default='LocalMax',
                        help='approach of estimations of doas from model output')
    parser.add_argument("--doa_probs_threshold", type=float, default=0.02)
    args = parser.parse_args()
    main(args)
