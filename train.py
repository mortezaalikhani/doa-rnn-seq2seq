"""
This module contains classes to train a doa estimation model with a given dataset.

"""

import argparse
import configparser
import csv
import math
import os
import pickle
import time

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
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


class TrainClassifier():
    """
    This class is for training a RNN classifier doa estimation model with a given dataset.

    """

    def train(self, model, iterator, optimizer, criterion, clip, device):
        model.train()

        epoch_loss = 0

        for i, batch in tqdm(enumerate(iterator)):
            src = batch[0]
            trg = batch[1]

            src = src.to(device)
            trg = trg.to(device)

            optimizer.zero_grad()

            output = model(src)

            loss = criterion(output, trg)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

            optimizer.step()

            epoch_loss += loss.item()

        return epoch_loss / len(iterator)

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

    def epoch_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    def train_model(self, model, train_dl, validation_dl, test_dl, optimizer, criterion, device, n_epochs, log_file,
                    model_save_address, config_file_address, kwargs, batch_size, doa_est_func, args,
                    roc_save_address=None, confusion_matrix_save_address=None):
        best_valid_loss = float('inf')
        best_epoch = 0

        train_loss = 0
        valid_loss = 0

        res_df = pd.DataFrame(
            columns=["epoch_number", "train_loss", "val_loss", "val_acc", "val_recall", "val_precision",
                     "val_f1", "val_support", "val_mae", "val_rmse"])

        writer = SummaryWriter()

        for epoch in range(n_epochs):

            start_time = time.time()

            print("train step ...")
            train_loss = self.train(model, train_dl, optimizer, criterion, 1, device)

            print("validation step ...")
            valid_loss, valid_acc, valid_recall, valid_precision, valid_f1, valid_support, valid_mae, valid_rmse, valid_auc = self.evaluate(
                model, validation_dl, criterion, device, doa_est_func)

            end_time = time.time()

            res_df = res_df.append(
                {"epoch_number": epoch, "train_loss": train_loss, "val_loss": valid_loss, "val_acc": valid_acc,
                 "val_recall": valid_recall, "val_precision": valid_precision, "val_f1": valid_f1,
                 "val_support": valid_support, "val_mae": valid_mae, "val_rmse": valid_rmse, "val_auc": valid_auc},
                ignore_index=True)

            epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_epoch = epoch
                torch.save(model.state_dict(), model_save_address)

            print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
            print(f'\t Val. acc: {valid_acc:.3f}')
            print(f'\t Val. recall: {valid_recall}')
            print(f'\t Val. precision: {valid_precision}')
            print(f'\t Val. f1: {valid_f1}')
            print(f'\t Val. support: {valid_support}')
            print(f'\t Val. mae: {valid_mae}')
            print(f'\t Val. rmse: {valid_rmse}')
            print(f'\t Val. auc: {valid_auc}')

            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/validation", valid_loss, epoch)
            writer.add_scalar("Accuracy/validation", valid_acc, epoch)
            writer.add_scalar("Recall/validation", valid_recall, epoch)
            writer.add_scalar("Precision/validation", valid_precision, epoch)
            writer.add_scalar("F1 score/validation", valid_f1, epoch)
            writer.add_scalar("Mae/validation", valid_mae, epoch)
            writer.add_scalar("Rmse/validation", valid_rmse, epoch)
            writer.add_scalar("Auc/validation", valid_auc, epoch)

        res_df.to_csv(log_file)

        model.load_state_dict(torch.load(model_save_address))

        print("test step ...")
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
        print(f'\t Best epoch: {best_epoch}')

        config = configparser.ConfigParser()
        config.read(config_file_address)
        id = str(len(config.sections()))
        config_dict = {}
        config_dict['working dir'] = args.working_dir
        config_dict['dataset dir'] = args.dataset_dir
        config_dict['input dim'] = str(kwargs['input_dim'])
        config_dict['dense size'] = str(kwargs['dense_size'])
        config_dict['hid dim'] = str(kwargs['hid_dim'])
        config_dict['num layer'] = str(kwargs['num_layer'])
        config_dict['num classes'] = str(kwargs['num_classes'])
        config_dict['dropout'] = str(kwargs['dropout'])
        config_dict['seed'] = str(kwargs['seed'])
        config_dict['num epochs'] = n_epochs
        config_dict['batch size'] = batch_size
        config_dict['learning rate'] = str(args.lr)
        config_dict['dataset type'] = args.dataset_type
        config_dict['model type'] = args.model_type
        config_dict['optimizer type'] = args.optimizer_type
        config_dict['loss type'] = args.loss_type
        config_dict['last epoch train loss'] = f"{train_loss:.3f}"
        config_dict['last epoch valid loss'] = f"{valid_loss:.3f}"
        config_dict['best epoch valid loss'] = f"{best_valid_loss:.3f}"
        config_dict['test loss'] = f"{test_loss:.3f}"
        config_dict['test acc'] = f"{test_acc:.3f}"
        config_dict['test recall'] = f"{test_recall}"
        config_dict['test precision'] = f"{test_precision}"
        config_dict['test f1'] = f"{test_f1}"
        config_dict['test support'] = f"{test_support}"
        config_dict['test mae'] = f"{test_mae}"
        config_dict['test rmse'] = f"{test_rmse}"
        config_dict['test auc'] = f"{test_auc}"
        config_dict['best epoch'] = f"{best_epoch}"
        config[id] = config_dict
        with open(config_file_address, 'w') as f:
            config.write(f)

        writer.close()


class TrainSeq2SeqNetwork():
    """
    This class is for training a Seq2Seq network doa estimation model with a given dataset.

    """

    def train(self, model, iterator, optimizer, criterion, clip, device, teacher_forcing_ratio):
        model.train()

        epoch_loss = 0

        for i, batch in tqdm(enumerate(iterator)):
            src = batch[0]
            trg = batch[1]

            src = src.to(device)
            trg = trg.to(device)

            optimizer.zero_grad()

            output = model(src, trg, teacher_forcing_ratio)

            # trg = [trg len, batch size]
            # output = [trg len, batch size, output dim]

            output_dim = output.shape[-1]

            output = output[1:].view(-1, output_dim)
            trg = torch.reshape(trg[1:], (-1,))

            # trg = [(trg len - 1) * batch size]
            # output = [(trg len - 1) * batch size, output dim]

            loss = criterion(output, trg)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

            optimizer.step()

            epoch_loss += loss.item()

        return epoch_loss / len(iterator)

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

    def epoch_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    def train_model(self, model, train_dl, validation_dl, test_dl, optimizer, criterion, device, n_epochs, log_file,
                    model_save_address, config_file_address, kwargs, batch_size, doa_est_func, args,
                    roc_save_address=None, confusion_matrix_save_address=None):
        best_valid_loss = float('inf')
        best_epoch = 0

        train_loss = 0
        valid_loss = 0

        res_df = pd.DataFrame(
            columns=["epoch_number", "train_loss", "val_loss", "val_acc", "val_recall", "val_precision",
                     "val_f1", "val_support", "val_mae", "val_rmse"])

        writer = SummaryWriter()

        for epoch in range(n_epochs):

            start_time = time.time()

            print("train step ...")
            train_loss = self.train(model, train_dl, optimizer, criterion, 1, device, args.teacher_forcing_ratio)

            print("validation step ...")
            valid_loss, valid_acc, valid_recall, valid_precision, valid_f1, valid_support, valid_mae, valid_rmse, valid_auc = self.evaluate(
                model, validation_dl, criterion, device, doa_est_func)

            end_time = time.time()

            res_df = res_df.append(
                {"epoch_number": epoch, "train_loss": train_loss, "val_loss": valid_loss, "val_acc": valid_acc,
                 "val_recall": valid_recall, "val_precision": valid_precision, "val_f1": valid_f1,
                 "val_support": valid_support, "val_mae": valid_mae, "val_rmse": valid_rmse, "val_auc": valid_auc},
                ignore_index=True)

            epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_epoch = epoch
                torch.save(model.state_dict(), model_save_address)

            print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
            print(f'\t Val. acc: {valid_acc:.3f}')
            print(f'\t Val. recall: {valid_recall}')
            print(f'\t Val. precision: {valid_precision}')
            print(f'\t Val. f1: {valid_f1}')
            print(f'\t Val. support: {valid_support}')
            print(f'\t Val. mae: {valid_mae}')
            print(f'\t Val. rmse: {valid_rmse}')
            print(f'\t Val. auc: {valid_auc}')

            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/validation", valid_loss, epoch)
            writer.add_scalar("Accuracy/validation", valid_acc, epoch)
            writer.add_scalar("Recall/validation", valid_recall, epoch)
            writer.add_scalar("Precision/validation", valid_precision, epoch)
            writer.add_scalar("F1 score/validation", valid_f1, epoch)
            writer.add_scalar("Mae/validation", valid_mae, epoch)
            writer.add_scalar("Rmse/validation", valid_rmse, epoch)
            writer.add_scalar("Auc/validation", valid_auc, epoch)

        res_df.to_csv(log_file)

        model.load_state_dict(torch.load(model_save_address))

        print("test step ...")
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
        print(f'\t Best epoch: {best_epoch}')

        config = configparser.ConfigParser()
        config.read(config_file_address)
        id = str(len(config.sections()))
        config_dict = {}
        config_dict['working dir'] = args.working_dir
        config_dict['dataset dir'] = args.dataset_dir
        config_dict['input dim'] = str(kwargs['input_dim'])
        config_dict['emb dim'] = str(kwargs['emb_dim'])
        config_dict['enc hid dim'] = str(kwargs['enc_hid_dim'])
        config_dict['dec hid dim'] = str(kwargs['dec_hid_dim'])
        config_dict['output dim'] = str(kwargs['output_dim'])
        config_dict['dropout'] = str(kwargs['dropout'])
        config_dict['enc num layers'] = str(kwargs['num_layers'])
        config_dict['seed'] = str(kwargs['seed'])
        config_dict['num epochs'] = n_epochs
        config_dict['batch size'] = batch_size
        config_dict['learning rate'] = str(args.lr)
        config_dict['dataset type'] = args.dataset_type
        config_dict['model type'] = args.model_type
        config_dict['optimizer type'] = args.optimizer_type
        config_dict['loss type'] = args.loss_type
        config_dict['teacher forcing ratio'] = str(args.teacher_forcing_ratio)
        config_dict['target len'] = str(args.trg_len)
        config_dict['last epoch train loss'] = f"{train_loss:.3f}"
        config_dict['last epoch valid loss'] = f"{valid_loss:.3f}"
        config_dict['best epoch valid loss'] = f"{best_valid_loss:.3f}"
        config_dict['test loss'] = f"{test_loss:.3f}"
        config_dict['test acc'] = f"{test_acc:.3f}"
        config_dict['test recall'] = f"{test_recall}"
        config_dict['test precision'] = f"{test_precision}"
        config_dict['test f1'] = f"{test_f1}"
        config_dict['test support'] = f"{test_support}"
        config_dict['test mae'] = f"{test_mae}"
        config_dict['test rmse'] = f"{test_rmse}"
        config_dict['test auc'] = f"{test_auc}"
        config_dict['best epoch'] = f"{best_epoch}"
        config[id] = config_dict
        with open(config_file_address, 'w') as f:
            config.write(f)

        writer.close()


def init_weights(model):
    for name, param in model.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main(args):
    os.makedirs(args.working_dir, exist_ok=True)

    train_set_file = os.path.join(args.dataset_dir, 'train.pkl')
    validation_set_file = os.path.join(args.dataset_dir, 'validation.pkl')
    test_set_file = os.path.join(args.dataset_dir, 'test.pkl')

    info_file = os.path.join(args.dataset_dir, 'info.pkl')
    with open(info_file, 'rb') as fp:
        info = pickle.load(fp)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device : {device}')

    if args.dataset_type == 'SequenceOfSnapshotsFixedN':
        train_ds = datasets.DatasetSequenceOfSnapshots(train_set_file)
        validation_ds = datasets.DatasetSequenceOfSnapshots(validation_set_file)
        test_ds = datasets.DatasetSequenceOfSnapshots(test_set_file)

        train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=datasets.sequence_of_snapshots_fixed_n_collate, pin_memory=True)
        validation_dl = DataLoader(validation_ds, batch_size=args.batch_size, shuffle=False,
                                   collate_fn=datasets.sequence_of_snapshots_fixed_n_collate, pin_memory=True)
        test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=datasets.sequence_of_snapshots_fixed_n_collate, pin_memory=True)

        kwargs = {'input_dim': 2 * info['M'][0],
                  'dense_size': args.dense_size,
                  'hid_dim': args.hidden_units,
                  'num_layer': args.num_layers,
                  'num_classes': max(info['doa_max']) - min(info['doa_min']),
                  'dropout': args.dropout,
                  'seed': 1234}
    elif args.dataset_type == 'SequenceOfSnapshotsVariableN':
        train_ds = datasets.DatasetSequenceOfSnapshotsSortedNumSnapshots(train_set_file)
        validation_ds = datasets.DatasetSequenceOfSnapshotsSortedNumSnapshots(validation_set_file)
        test_ds = datasets.DatasetSequenceOfSnapshotsSortedNumSnapshots(test_set_file)

        train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                              collate_fn=datasets.sequence_of_snapshots_variable_n_collate, pin_memory=True)
        validation_dl = DataLoader(validation_ds, batch_size=args.batch_size, shuffle=False,
                                   collate_fn=datasets.sequence_of_snapshots_variable_n_collate, pin_memory=True)
        test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=datasets.sequence_of_snapshots_variable_n_collate, pin_memory=True)

        kwargs = {'input_dim': 2 * info['M'][0],
                  'dense_size': args.dense_size,
                  'hid_dim': args.hidden_units,
                  'num_layer': args.num_layers,
                  'num_classes': max(info['doa_max']) - min(info['doa_min']),
                  'dropout': args.dropout,
                  'seed': 1234}
    elif args.dataset_type == 'SequenceOfAntennas':
        train_ds = datasets.DatasetSequenceOfAntennas(train_set_file)
        validation_ds = datasets.DatasetSequenceOfAntennas(validation_set_file)
        test_ds = datasets.DatasetSequenceOfAntennas(test_set_file)

        train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=datasets.sequence_of_antennas_collate, pin_memory=True)
        validation_dl = DataLoader(validation_ds, batch_size=args.batch_size, shuffle=False,
                                   collate_fn=datasets.sequence_of_antennas_collate, pin_memory=True)
        test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=datasets.sequence_of_antennas_collate, pin_memory=True)

        kwargs = {'input_dim': info['N'][0],
                  'dense_size': args.dense_size,
                  'hid_dim': args.hidden_units,
                  'num_layer': args.num_layers,
                  'num_classes': max(info['doa_max']) - min(info['doa_min']),
                  'dropout': args.dropout,
                  'seed': 1234}
    elif args.dataset_type == 'SequenceOfCovMatrixRows':
        train_ds = datasets.DatasetSequenceOfCovMatrixRows(train_set_file)
        validation_ds = datasets.DatasetSequenceOfCovMatrixRows(validation_set_file)
        test_ds = datasets.DatasetSequenceOfCovMatrixRows(test_set_file)

        train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=datasets.sequence_of_cov_matrix_rows_collate, pin_memory=True)
        validation_dl = DataLoader(validation_ds, batch_size=args.batch_size, shuffle=False,
                                   collate_fn=datasets.sequence_of_cov_matrix_rows_collate, pin_memory=True)
        test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=datasets.sequence_of_cov_matrix_rows_collate, pin_memory=True)

        kwargs = {'input_dim': 2 * info['M'][0],
                  'dense_size': args.dense_size,
                  'hid_dim': args.hidden_units,
                  'num_layer': args.num_layers,
                  'num_classes': max(info['doa_max']) - min(info['doa_min']),
                  'dropout': args.dropout,
                  'seed': 1234}
    elif args.dataset_type == 'SequenceOfSnapshotsFixedNForSeq2Seq':
        train_ds = datasets.DatasetSequenceOfSnapshotsForSeq2Seq(train_set_file, args.trg_len)
        validation_ds = datasets.DatasetSequenceOfSnapshotsForSeq2Seq(validation_set_file, args.trg_len)
        test_ds = datasets.DatasetSequenceOfSnapshotsForSeq2Seq(test_set_file, args.trg_len)

        train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=datasets.sequence_of_snapshots_fixed_n_for_seq2seq_collate, pin_memory=True)
        validation_dl = DataLoader(validation_ds, batch_size=args.batch_size, shuffle=False,
                                   collate_fn=datasets.sequence_of_snapshots_fixed_n_for_seq2seq_collate,
                                   pin_memory=True)
        test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=datasets.sequence_of_snapshots_fixed_n_for_seq2seq_collate, pin_memory=True)

        kwargs = {'input_dim': 2 * info['M'][0],
                  'emb_dim': args.dense_size,
                  'enc_hid_dim': args.hidden_units,
                  'dec_hid_dim': args.hidden_units_dec,
                  'dropout': args.dropout,
                  'num_layers': args.num_layers,
                  'seed': 1234,
                  'output_dim': max(info['doa_max']) - min(info['doa_min']) + 2}
        kwargs_encoder = {'input_dim': 2 * info['M'][0],
                          'emb_dim': args.dense_size,
                          'enc_hid_dim': args.hidden_units,
                          'dec_hid_dim': args.hidden_units_dec,
                          'dropout': args.dropout,
                          'num_layers': args.num_layers,
                          'seed': 1234}
        kwargs_attention = {'enc_hid_dim': args.hidden_units,
                            'dec_hid_dim': args.hidden_units_dec,
                            'seed': 1234}
        kwargs_decoder = {'output_dim': max(info['doa_max']) - min(info['doa_min']) + 2,
                          'emb_dim': args.dense_size,
                          'enc_hid_dim': args.hidden_units,
                          'dec_hid_dim': args.hidden_units_dec,
                          'dropout': args.dropout,
                          'seed': 1234}
    elif args.dataset_type == 'SequenceOfSnapshotsVariableNForSeq2Seq':
        train_ds = datasets.DatasetSequenceOfSnapshotsForSeq2SeqSortedNumSnapshots(train_set_file, args.trg_len)
        validation_ds = datasets.DatasetSequenceOfSnapshotsForSeq2SeqSortedNumSnapshots(validation_set_file,
                                                                                        args.trg_len)
        test_ds = datasets.DatasetSequenceOfSnapshotsForSeq2SeqSortedNumSnapshots(test_set_file, args.trg_len)

        train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=datasets.sequence_of_snapshots_variable_n_for_seq2seq_collate, pin_memory=True)
        validation_dl = DataLoader(validation_ds, batch_size=args.batch_size, shuffle=False,
                                   collate_fn=datasets.sequence_of_snapshots_variable_n_for_seq2seq_collate,
                                   pin_memory=True)
        test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=datasets.sequence_of_snapshots_variable_n_for_seq2seq_collate, pin_memory=True)

        kwargs = {'input_dim': 2 * info['M'][0],
                  'emb_dim': args.dense_size,
                  'enc_hid_dim': args.hidden_units,
                  'dec_hid_dim': args.hidden_units_dec,
                  'dropout': args.dropout,
                  'num_layers': args.num_layers,
                  'seed': 1234,
                  'output_dim': max(info['doa_max']) - min(info['doa_min']) + 2}
        kwargs_encoder = {'input_dim': 2 * info['M'][0],
                          'emb_dim': args.dense_size,
                          'enc_hid_dim': args.hidden_units,
                          'dec_hid_dim': args.hidden_units_dec,
                          'dropout': args.dropout,
                          'num_layers': args.num_layers,
                          'seed': 1234}
        kwargs_attention = {'enc_hid_dim': args.hidden_units,
                            'dec_hid_dim': args.hidden_units_dec,
                            'seed': 1234}
        kwargs_decoder = {'output_dim': max(info['doa_max']) - min(info['doa_min']) + 2,
                          'emb_dim': args.dense_size,
                          'enc_hid_dim': args.hidden_units,
                          'dec_hid_dim': args.hidden_units_dec,
                          'dropout': args.dropout,
                          'seed': 1234}
    elif args.dataset_type == 'SequenceOfCovMatrixRowsForSeq2Seq':
        train_ds = datasets.DatasetSequenceOfCovMatrixRowsForSeq2Seq(train_set_file, args.trg_len)
        validation_ds = datasets.DatasetSequenceOfCovMatrixRowsForSeq2Seq(validation_set_file, args.trg_len)
        test_ds = datasets.DatasetSequenceOfCovMatrixRowsForSeq2Seq(test_set_file, args.trg_len)

        train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=datasets.sequence_of_cov_matrix_rows_for_seq2seq_collate, pin_memory=True)
        validation_dl = DataLoader(validation_ds, batch_size=args.batch_size, shuffle=False,
                                   collate_fn=datasets.sequence_of_cov_matrix_rows_for_seq2seq_collate, pin_memory=True)
        test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=datasets.sequence_of_cov_matrix_rows_for_seq2seq_collate, pin_memory=True)

        kwargs = {'input_dim': 2 * info['M'][0],
                  'emb_dim': args.dense_size,
                  'enc_hid_dim': args.hidden_units,
                  'dec_hid_dim': args.hidden_units_dec,
                  'dropout': args.dropout,
                  'num_layers': args.num_layers,
                  'seed': 1234,
                  'output_dim': max(info['doa_max']) - min(info['doa_min']) + 2}
        kwargs_encoder = {'input_dim': 2 * info['M'][0],
                          'emb_dim': args.dense_size,
                          'enc_hid_dim': args.hidden_units,
                          'dec_hid_dim': args.hidden_units_dec,
                          'dropout': args.dropout,
                          'num_layers': args.num_layers,
                          'seed': 1234}
        kwargs_attention = {'enc_hid_dim': args.hidden_units,
                            'dec_hid_dim': args.hidden_units_dec,
                            'seed': 1234}
        kwargs_decoder = {'output_dim': max(info['doa_max']) - min(info['doa_min']) + 2,
                          'emb_dim': args.dense_size,
                          'enc_hid_dim': args.hidden_units,
                          'dec_hid_dim': args.hidden_units_dec,
                          'dropout': args.dropout,
                          'seed': 1234}
    else:
        print("The value for the dataset_type argument is invalid")
        exit()

    if args.model_type == 'BILSTM':
        model = BILSTMClassifier(**kwargs)
    elif args.model_type == 'LSTM':
        model = LSTMClassifier(**kwargs)
    elif args.model_type == 'BIGRU':
        model = BIGRUClassifier(**kwargs)
    elif args.model_type == 'GRU':
        model = GRUClassifier(**kwargs)
    elif args.model_type == 'Seq2Seq':
        encoder = Encoder(**kwargs_encoder)
        attention = Attention(**kwargs_attention)
        decoder = Decoder(attention=attention, **kwargs_decoder)
        model = Seq2Seq(encoder, decoder, device)
    else:
        print("The value for the model_type argument is invalid")
        exit()

    model_save_address = os.path.join(args.working_dir, 'model.pt')
    log_file = os.path.join(args.working_dir, 'train.csv')
    config_file_address = args.config_path

    params_file = os.path.join(args.working_dir, 'params.csv')
    print('writing {}...'.format(params_file))
    with open(params_file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['keys', 'values'])
        for arg in vars(args):
            writer.writerow([arg, getattr(args, arg)])
        writer.writerow(['input_dim', kwargs['input_dim']])
        if args.dataset_type in ['SequenceOfSnapshotsFixedNForSeq2Seq', 'SequenceOfSnapshotsVariableNForSeq2Seq',
                                 'SequenceOfCovMatrixRowsForSeq2Seq']:
            writer.writerow(['output_dim', kwargs['output_dim']])
        else:
            writer.writerow(['num_classes', kwargs['num_classes']])
        writer.writerow(['seed', kwargs['seed']])

    roc_save_address = os.path.join(args.working_dir, 'roc_curve.png')
    confusion_matrix_save_address = os.path.join(args.working_dir, 'confusion_matrix.png')

    model = model.to(device)
    model.apply(init_weights)
    print(f'The model has {count_parameters(model):,} trainable parameters')

    if args.optimizer_type == 'ADAM':
        if args.lr is None:
            optimizer = optim.Adam(model.parameters())
        else:
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer_type == 'SGD':
        if args.lr is None:
            print(f"The value for the learning rate could not be None for the {args.optimizer_type} optimizer")
            exit()
        else:
            optimizer = optim.SGD(model.parameters(), lr=args.lr)
    else:
        print("The value for the optimizer_type argument is invalid")
        exit()

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

    if args.model_type == 'Seq2Seq':
        ts_instance = TrainSeq2SeqNetwork()
        ts_instance.train_model(model, train_dl, validation_dl, test_dl, optimizer, criterion, device, args.epochs,
                                log_file, model_save_address, config_file_address, kwargs, args.batch_size,
                                doa_est_func,
                                args, roc_save_address=roc_save_address,
                                confusion_matrix_save_address=confusion_matrix_save_address)
    else:
        tc_instance = TrainClassifier()
        tc_instance.train_model(model, train_dl, validation_dl, test_dl, optimizer, criterion, device, args.epochs,
                                log_file, model_save_address, config_file_address, kwargs, args.batch_size,
                                doa_est_func,
                                args, roc_save_address=roc_save_address,
                                confusion_matrix_save_address=confusion_matrix_save_address)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, help='Directory of dataset')
    parser.add_argument('-l', '--num_layers', type=int, default=1, help='The number of RNN layers')
    parser.add_argument('-u', '--hidden_units', type=int, default=256,
                        help='The number of hidden RNN units')
    parser.add_argument('-d', '--dropout', type=float, default=0.5, help='The dropout percentage')
    parser.add_argument('-s', '--dense_size', type=int, default=256, help='Size of the dense layer')
    parser.add_argument('-e', '--epochs', type=int, default=200, help='The number of epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=16, help='The batch size')
    parser.add_argument('-r', '--lr', type=float, default=None, help='The learning rate')
    parser.add_argument('-c', '--hidden_units_dec', type=int, default=256,
                        help='The number of hidden RNN units in decoder')
    parser.add_argument('--trg_len', type=int, default=6,
                        help='Target length in Seq2Seq network')
    parser.add_argument('--teacher_forcing_ratio', type=float, default=0.5,
                        help='Teacher forcing ratio in Seq2Seq network')
    parser.add_argument('--working_dir', type=str, help='Where to save checkpoints and logs')
    parser.add_argument('--dataset_type', type=str,
                        choices=['SequenceOfSnapshotsFixedN', 'SequenceOfSnapshotsVariableN', 'SequenceOfAntennas',
                                 'SequenceOfCovMatrixRows', 'SequenceOfSnapshotsFixedNForSeq2Seq',
                                 'SequenceOfSnapshotsVariableNForSeq2Seq', 'SequenceOfCovMatrixRowsForSeq2Seq'],
                        default='SequenceOfSnapshotsFixedN', help='dataset type')
    parser.add_argument('--model_type', type=str, choices=['BILSTM', 'LSTM', 'BIGRU', 'GRU', 'Seq2Seq'],
                        default='BILSTM',
                        help='model type')
    parser.add_argument('--optimizer_type', type=str, choices=['ADAM', 'SGD'], default='ADAM', help='optimizer type')
    parser.add_argument('--loss_type', type=str, choices=['BCE', 'FOCAL', 'CrossEntropy'], default='BCE',
                        help='loss function type')
    parser.add_argument('--estimation_of_doas', type=str, choices=['MaxK', 'LocalMaxK', 'Seq2Seq'], default='LocalMaxK',
                        help='approach of estimations of doas from model output')
    parser.add_argument('--config_path', type=str, default='config.ini', help='config file path')
    args = parser.parse_args()
    main(args)
