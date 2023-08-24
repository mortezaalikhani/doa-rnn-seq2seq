"""
This module contains classes to evaluate a doa estimation model with a given test dataset.

"""

import argparse
import json
import math
import os

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from net.Attention import Attention
from net.BIGRUClassifier import BIGRUClassifier
from net.BILSTMClassifier import BILSTMClassifier
from net.Decoder import Decoder
from net.Encoder import Encoder
from net.GRUClassifier import GRUClassifier
from net.LSTMClassifier import LSTMClassifier
from net.Seq2Seq import Seq2Seq
from utils import utils, datasets


class EvaluateClassifier():
    """
    This class is for evaluating a RNN classifier doa estimation model with a given test dataset.

    """

    def evaluate(self, model, iterator, criterion, device, doa_est_func, plot_roc=False, roc_save_address=None,
                 plot_confusion_matrix=False, confusion_matrix_save_address=None):
        model.eval()

        epoch_loss = 0
        probs_tensor_list = []
        outputs_tensor_list = []
        targets_tensor_list = []
        estimated_doas_list = []
        with torch.no_grad():
            for i, batch in tqdm(enumerate(iterator)):
                src = batch[0]
                trg = batch[1]

                src = src.to(device)
                trg = trg.to(device)

                output = model(src)

                loss = criterion(output, trg)

                epoch_loss += loss.item()

                output = output.to('cpu')
                trg = trg.to('cpu')

                probs_tensor_list.append(output)

                output, estimated_doas = doa_est_func(output, trg)

                estimated_doas_list.extend(estimated_doas)

                outputs_tensor_list.append(output)
                targets_tensor_list.append(trg)

        probs_tensor = torch.vstack(probs_tensor_list)
        outputs_tensor = torch.vstack(outputs_tensor_list)
        targets_tensor = torch.vstack(targets_tensor_list)

        targets_list = torch.reshape(targets_tensor, [-1]).tolist()
        outputs_list = torch.reshape(outputs_tensor, [-1]).tolist()
        probs_list = torch.reshape(probs_tensor, [-1]).tolist()
        acc, recall, precision, f1, support = utils.cal_eval_metrics(outputs_list, targets_list)
        doa_errors = utils.cal_doa_errors(estimated_doas_list, targets_tensor)
        mae = utils.cal_mae(doa_errors)
        rmse = utils.cal_rmse(doa_errors)
        auc = utils.cal_auc(targets_list, probs_list)
        if plot_roc:
            utils.plot_roc_curve(targets_list, probs_list, roc_save_address)
        if plot_confusion_matrix:
            utils.plot_confusion_matrix(targets_list, outputs_list, save_address=confusion_matrix_save_address)

        return epoch_loss / len(iterator), acc, recall, precision, f1, support, mae, rmse, auc

    def evaluate_model(self, model, test_dl, criterion, device, doa_est_func, out_dir):
        roc_save_address = os.path.join(out_dir, "roc_curve.png")
        confusion_matrix_save_address = os.path.join(out_dir, "confusion_matrix.png")
        res_save_address = os.path.join(out_dir, "results.json")

        test_loss, test_acc, test_recall, test_precision, test_f1, test_support, test_mae, test_rmse, test_auc = self.evaluate(
            model, test_dl, criterion, device, doa_est_func, plot_roc=True, roc_save_address=roc_save_address,
            plot_confusion_matrix=True, confusion_matrix_save_address=confusion_matrix_save_address)

        print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
        print(f'\t Test. acc: {test_acc:.3f}')
        print(f'\t Test. recall: {test_recall}')
        print(f'\t Test. precision: {test_precision}')
        print(f'\t Test. f1: {test_f1}')
        print(f'\t Test. support: {test_support}')
        print(f'\t Test. mae: {test_mae}')
        print(f'\t Test. rmse: {test_rmse}')
        print(f'\t Test. auc: {test_auc}')

        res_dict = {"test_loss": test_loss, "test_acc": test_acc, "test_recall": test_recall,
                    "test_precision": test_precision, "test_f1": test_f1, "test_support": test_support,
                    "test_mae": test_mae, "test_rmse": test_rmse, "test_auc": test_auc}

        with open(res_save_address, "w") as f:
            json.dump(res_dict, f)


class EvaluateSeq2SeqNetwork():
    """
    This class is for evaluating a Seq2Seq network doa estimation model with a given test dataset.

    """

    def evaluate(self, model, iterator, criterion, device, doa_est_func, plot_roc=False, roc_save_address=None,
                 plot_confusion_matrix=False, confusion_matrix_save_address=None):
        model.eval()

        epoch_loss = 0
        probs_tensor_list = []
        outputs_tensor_list = []
        targets_tensor_list = []
        estimated_doas_list = []
        with torch.no_grad():
            for i, batch in tqdm(enumerate(iterator)):
                src = batch[0]
                trg = batch[1]

                src = src.to(device)
                trg = trg.to(device)

                output = model(src, trg, 0)

                # trg = [trg len, batch size]
                # output = [trg len, batch size, output dim]

                output_dim = output.shape[-1]

                output_flatten = output[1:].view(-1, output_dim)
                trg_flatten = torch.reshape(trg[1:], (-1,))

                # trg = [(trg len - 1) * batch size]
                # output = [(trg len - 1) * batch size, output dim]

                loss = criterion(output_flatten, trg_flatten)

                epoch_loss += loss.item()

                output = output.to('cpu')
                trg = trg.to('cpu')

                output, estimated_doas, trg, prob = doa_est_func(output, trg)

                estimated_doas_list.extend(estimated_doas)

                outputs_tensor_list.append(output)
                targets_tensor_list.append(trg)
                probs_tensor_list.append(prob)

        probs_tensor = torch.vstack(probs_tensor_list)
        outputs_tensor = torch.vstack(outputs_tensor_list)
        targets_tensor = torch.vstack(targets_tensor_list)

        targets_list = torch.reshape(targets_tensor, [-1]).tolist()
        outputs_list = torch.reshape(outputs_tensor, [-1]).tolist()
        probs_list = torch.reshape(probs_tensor, [-1]).tolist()
        acc, recall, precision, f1, support = utils.cal_eval_metrics(outputs_list, targets_list)
        doa_errors = utils.cal_doa_errors(estimated_doas_list, targets_tensor)
        mae = utils.cal_mae(doa_errors)
        rmse = utils.cal_rmse(doa_errors)
        auc = utils.cal_auc(targets_list, probs_list)
        if plot_roc:
            utils.plot_roc_curve(targets_list, probs_list, roc_save_address)
        if plot_confusion_matrix:
            utils.plot_confusion_matrix(targets_list, outputs_list, save_address=confusion_matrix_save_address)

        return epoch_loss / len(iterator), acc, recall, precision, f1, support, mae, rmse, auc

    def evaluate_model(self, model, test_dl, criterion, device, doa_est_func, out_dir):
        roc_save_address = os.path.join(out_dir, "roc_curve.png")
        confusion_matrix_save_address = os.path.join(out_dir, "confusion_matrix.png")
        res_save_address = os.path.join(out_dir, "results.json")

        test_loss, test_acc, test_recall, test_precision, test_f1, test_support, test_mae, test_rmse, test_auc = self.evaluate(
            model, test_dl, criterion, device, doa_est_func, plot_roc=True, roc_save_address=roc_save_address,
            plot_confusion_matrix=True, confusion_matrix_save_address=confusion_matrix_save_address)

        print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
        print(f'\t Test. acc: {test_acc:.3f}')
        print(f'\t Test. recall: {test_recall}')
        print(f'\t Test. precision: {test_precision}')
        print(f'\t Test. f1: {test_f1}')
        print(f'\t Test. support: {test_support}')
        print(f'\t Test. mae: {test_mae}')
        print(f'\t Test. rmse: {test_rmse}')
        print(f'\t Test. auc: {test_auc}')

        res_dict = {"test_loss": test_loss, "test_acc": test_acc, "test_recall": test_recall,
                    "test_precision": test_precision, "test_f1": test_f1, "test_support": test_support,
                    "test_mae": test_mae, "test_rmse": test_rmse, "test_auc": test_auc}

        with open(res_save_address, "w") as f:
            json.dump(res_dict, f)


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device : {device}')

    model_params_address = os.path.join(args.model_dir, "params.csv")
    params_df = pd.read_csv(model_params_address)

    if args.dataset_type == 'SequenceOfSnapshotsFixedN':
        test_ds = datasets.DatasetSequenceOfSnapshots(args.test_dataset_path)

        test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=datasets.sequence_of_snapshots_fixed_n_collate, pin_memory=True)
        kwargs = {'input_dim': int(params_df.loc[params_df['keys'] == 'input_dim', 'values']),
                  'dense_size': int(params_df.loc[params_df['keys'] == 'dense_size', 'values']),
                  'hid_dim': int(params_df.loc[params_df['keys'] == 'hidden_units', 'values']),
                  'num_layer': int(params_df.loc[params_df['keys'] == 'num_layers', 'values']),
                  'num_classes': int(params_df.loc[params_df['keys'] == 'num_classes', 'values']),
                  'dropout': float(params_df.loc[params_df['keys'] == 'dropout', 'values']),
                  'seed': int(params_df.loc[params_df['keys'] == 'seed', 'values'])}
    elif args.dataset_type == 'SequenceOfSnapshotsVariableN':
        test_ds = datasets.DatasetSequenceOfSnapshotsSortedNumSnapshots(args.test_dataset_path)

        test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=datasets.sequence_of_snapshots_variable_n_collate, pin_memory=True)
        kwargs = {'input_dim': int(params_df.loc[params_df['keys'] == 'input_dim', 'values']),
                  'dense_size': int(params_df.loc[params_df['keys'] == 'dense_size', 'values']),
                  'hid_dim': int(params_df.loc[params_df['keys'] == 'hidden_units', 'values']),
                  'num_layer': int(params_df.loc[params_df['keys'] == 'num_layers', 'values']),
                  'num_classes': int(params_df.loc[params_df['keys'] == 'num_classes', 'values']),
                  'dropout': float(params_df.loc[params_df['keys'] == 'dropout', 'values']),
                  'seed': int(params_df.loc[params_df['keys'] == 'seed', 'values'])}
    elif args.dataset_type == 'SequenceOfAntennas':
        test_ds = datasets.DatasetSequenceOfAntennas(args.test_dataset_path)

        test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=datasets.sequence_of_antennas_collate, pin_memory=True)
        kwargs = {'input_dim': int(params_df.loc[params_df['keys'] == 'input_dim', 'values']),
                  'dense_size': int(params_df.loc[params_df['keys'] == 'dense_size', 'values']),
                  'hid_dim': int(params_df.loc[params_df['keys'] == 'hidden_units', 'values']),
                  'num_layer': int(params_df.loc[params_df['keys'] == 'num_layers', 'values']),
                  'num_classes': int(params_df.loc[params_df['keys'] == 'num_classes', 'values']),
                  'dropout': float(params_df.loc[params_df['keys'] == 'dropout', 'values']),
                  'seed': int(params_df.loc[params_df['keys'] == 'seed', 'values'])}
    elif args.dataset_type == 'SequenceOfCovMatrixRows':
        test_ds = datasets.DatasetSequenceOfCovMatrixRows(args.test_dataset_path)

        test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=datasets.sequence_of_cov_matrix_rows_collate, pin_memory=True)
        kwargs = {'input_dim': int(params_df.loc[params_df['keys'] == 'input_dim', 'values']),
                  'dense_size': int(params_df.loc[params_df['keys'] == 'dense_size', 'values']),
                  'hid_dim': int(params_df.loc[params_df['keys'] == 'hidden_units', 'values']),
                  'num_layer': int(params_df.loc[params_df['keys'] == 'num_layers', 'values']),
                  'num_classes': int(params_df.loc[params_df['keys'] == 'num_classes', 'values']),
                  'dropout': float(params_df.loc[params_df['keys'] == 'dropout', 'values']),
                  'seed': int(params_df.loc[params_df['keys'] == 'seed', 'values'])}
    elif args.dataset_type == 'SequenceOfSnapshotsFixedNForSeq2Seq':
        test_ds = datasets.DatasetSequenceOfSnapshotsForSeq2Seq(args.test_dataset_path,
                                                                int(params_df.loc[
                                                                        params_df['keys'] == 'trg_len', 'values']))

        test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=datasets.sequence_of_snapshots_fixed_n_for_seq2seq_collate, pin_memory=True)
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
    elif args.dataset_type == 'SequenceOfSnapshotsVariableNForSeq2Seq':
        test_ds = datasets.DatasetSequenceOfSnapshotsForSeq2SeqSortedNumSnapshots(args.test_dataset_path,
                                                                                  int(params_df.loc[params_df[
                                                                                                        'keys'] == 'trg_len', 'values']))

        test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=datasets.sequence_of_snapshots_variable_n_for_seq2seq_collate, pin_memory=True)
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
        test_ds = datasets.DatasetSequenceOfCovMatrixRowsForSeq2Seq(args.test_dataset_path,
                                                                    int(params_df.loc[
                                                                            params_df['keys'] == 'trg_len', 'values']))

        test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=datasets.sequence_of_cov_matrix_rows_for_seq2seq_collate, pin_memory=True)
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
    elif args.estimation_of_doas == 'LocalMaxK':
        doa_est_func = utils.estimate_doas_local_max_k
    elif args.estimation_of_doas == 'Seq2Seq':
        doa_est_func = utils.estimate_doas_seq2seq
    else:
        print("The value for the estimation_of_doas argument is invalid")
        exit()

    if model_type == 'Seq2Seq':
        ts_instance = EvaluateSeq2SeqNetwork()
        ts_instance.evaluate_model(model, test_dl, criterion, device, doa_est_func, args.out_dir)
    else:
        tc_instance = EvaluateClassifier()
        tc_instance.evaluate_model(model, test_dl, criterion, device, doa_est_func, args.out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dataset_path', type=str, help='Path of test dataset')
    parser.add_argument('--model_dir', type=str, help='Directory of model')
    parser.add_argument('--out_dir', type=str, help='Path for directory that results will be saved in it')
    parser.add_argument('-b', '--batch_size', type=int, default=16, help='The batch size')
    parser.add_argument('--dataset_type', type=str,
                        choices=['SequenceOfSnapshotsFixedN', 'SequenceOfSnapshotsVariableN', 'SequenceOfAntennas',
                                 'SequenceOfCovMatrixRows', 'SequenceOfSnapshotsFixedNForSeq2Seq',
                                 'SequenceOfSnapshotsVariableNForSeq2Seq', 'SequenceOfCovMatrixRowsForSeq2Seq'],
                        default='SequenceOfSnapshotsFixedN', help='dataset type')
    parser.add_argument('--loss_type', type=str, choices=['BCE', 'FOCAL', 'CrossEntropy'], default='BCE',
                        help='loss function type')
    parser.add_argument('--estimation_of_doas', type=str, choices=['MaxK', 'LocalMaxK', 'Seq2Seq'], default='LocalMaxK',
                        help='approach of estimations of doas from model output')
    args = parser.parse_args()
    main(args)
