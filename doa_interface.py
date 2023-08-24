"""
This module contains a class that is a interface for evaluating RNN classfier and Seq2Seq network models and comparing them with other methods of doa estimation.

"""

import os
import time

import pandas as pd
import torch
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


class DoaInterface():
    """
    This class is a interface for evaluating RNN classfier and Seq2Seq network models and comparing them with other methods of doa estimation.

    """

    def __init__(self, model_dir, dataset_type, estimation_of_doas, batch_size, device):
        self.dataset_type = dataset_type
        self.batch_size = batch_size
        self.device = device

        model_params_address = os.path.join(model_dir, "params.csv")
        params_df = pd.read_csv(model_params_address)

        if self.dataset_type == 'SequenceOfSnapshotsFixedN':
            kwargs = {'input_dim': int(params_df.loc[params_df['keys'] == 'input_dim', 'values']),
                      'dense_size': int(params_df.loc[params_df['keys'] == 'dense_size', 'values']),
                      'hid_dim': int(params_df.loc[params_df['keys'] == 'hidden_units', 'values']),
                      'num_layer': int(params_df.loc[params_df['keys'] == 'num_layers', 'values']),
                      'num_classes': int(params_df.loc[params_df['keys'] == 'num_classes', 'values']),
                      'dropout': float(params_df.loc[params_df['keys'] == 'dropout', 'values']),
                      'seed': int(params_df.loc[params_df['keys'] == 'seed', 'values'])}
        elif self.dataset_type == 'SequenceOfSnapshotsVariableN':
            kwargs = {'input_dim': int(params_df.loc[params_df['keys'] == 'input_dim', 'values']),
                      'dense_size': int(params_df.loc[params_df['keys'] == 'dense_size', 'values']),
                      'hid_dim': int(params_df.loc[params_df['keys'] == 'hidden_units', 'values']),
                      'num_layer': int(params_df.loc[params_df['keys'] == 'num_layers', 'values']),
                      'num_classes': int(params_df.loc[params_df['keys'] == 'num_classes', 'values']),
                      'dropout': float(params_df.loc[params_df['keys'] == 'dropout', 'values']),
                      'seed': int(params_df.loc[params_df['keys'] == 'seed', 'values'])}
        elif self.dataset_type == 'SequenceOfAntennas':
            kwargs = {'input_dim': int(params_df.loc[params_df['keys'] == 'input_dim', 'values']),
                      'dense_size': int(params_df.loc[params_df['keys'] == 'dense_size', 'values']),
                      'hid_dim': int(params_df.loc[params_df['keys'] == 'hidden_units', 'values']),
                      'num_layer': int(params_df.loc[params_df['keys'] == 'num_layers', 'values']),
                      'num_classes': int(params_df.loc[params_df['keys'] == 'num_classes', 'values']),
                      'dropout': float(params_df.loc[params_df['keys'] == 'dropout', 'values']),
                      'seed': int(params_df.loc[params_df['keys'] == 'seed', 'values'])}
        elif self.dataset_type == 'SequenceOfCovMatrixRows':
            kwargs = {'input_dim': int(params_df.loc[params_df['keys'] == 'input_dim', 'values']),
                      'dense_size': int(params_df.loc[params_df['keys'] == 'dense_size', 'values']),
                      'hid_dim': int(params_df.loc[params_df['keys'] == 'hidden_units', 'values']),
                      'num_layer': int(params_df.loc[params_df['keys'] == 'num_layers', 'values']),
                      'num_classes': int(params_df.loc[params_df['keys'] == 'num_classes', 'values']),
                      'dropout': float(params_df.loc[params_df['keys'] == 'dropout', 'values']),
                      'seed': int(params_df.loc[params_df['keys'] == 'seed', 'values'])}
        elif self.dataset_type == 'SequenceOfSnapshotsFixedNForSeq2Seq':
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
            self.trg_len = int(params_df.loc[params_df['keys'] == 'trg_len', 'values'])
        elif self.dataset_type == 'SequenceOfSnapshotsVariableNForSeq2Seq':
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
            self.trg_len = int(params_df.loc[params_df['keys'] == 'trg_len', 'values'])
        elif self.dataset_type == 'SequenceOfCovMatrixRowsForSeq2Seq':
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
            self.trg_len = int(params_df.loc[params_df['keys'] == 'trg_len', 'values'])
        else:
            print("The value for the dataset_type argument is invalid")
            exit()

        self.model_type = list(params_df.loc[params_df['keys'] == 'model_type', 'values'])[0]
        if self.model_type == 'BILSTM':
            self.model = BILSTMClassifier(**kwargs)
        elif self.model_type == 'LSTM':
            self.model = LSTMClassifier(**kwargs)
        elif self.model_type == 'BIGRU':
            self.model = BIGRUClassifier(**kwargs)
        elif self.model_type == 'GRU':
            self.model = GRUClassifier(**kwargs)
        elif self.model_type == 'Seq2Seq':
            encoder = Encoder(**kwargs_encoder)
            attention = Attention(**kwargs_attention)
            decoder = Decoder(attention=attention, **kwargs_decoder)
            self.model = Seq2Seq(encoder, decoder, device)
        else:
            print("The value for the model_type argument is invalid")
            exit()

        model_save_address = os.path.join(model_dir, 'model.pt')
        self.model = self.model.to(device)
        self.model.load_state_dict(torch.load(model_save_address))
        self.model.eval()

        if estimation_of_doas == 'MaxK':
            self.doa_est_func = utils.estimate_doas_max_k
        elif estimation_of_doas == 'LocalMaxK':
            self.doa_est_func = utils.estimate_doas_local_max_k
        elif estimation_of_doas == 'Seq2Seq':
            self.doa_est_func = utils.estimate_doas_seq2seq
        else:
            print("The value for the estimation_of_doas argument is invalid")
            exit()

    def doa_interface(self, dataset):
        sample_info_list = [sample["sample_info"] for sample in dataset]
        t_start = time.time()
        dl = self.get_dataloader(dataset)
        if self.model_type == 'Seq2Seq':
            estimated_doas_list, targets_tensor = self.predict_seq2seq_network(dl)
        else:
            estimated_doas_list, targets_tensor = self.predict_classifier(dl)
        t_end = time.time()
        doa_errors_list = utils.cal_doa_errors_for_samples(estimated_doas_list, targets_tensor)
        output = {"output_list": [], "total_time": (t_end - t_start)}
        if self.dataset_type in ["SequenceOfSnapshotsVariableN", "SequenceOfSnapshotsVariableNForSeq2Seq"]:
            sample_info_list_sorted = list(sorted(sample_info_list, key=lambda x: x['N'], reverse=True))
            for i, sample_info in enumerate(sample_info_list_sorted):
                output["output_list"].append({"sample_info": sample_info, "doa_errors": doa_errors_list[i]})
        else:
            for i, sample_info in enumerate(sample_info_list):
                output["output_list"].append({"sample_info": sample_info, "doa_errors": doa_errors_list[i]})
        return output

    def get_dataloader(self, dataset):
        if self.dataset_type == 'SequenceOfSnapshotsFixedN':
            ds = datasets.DatasetSequenceOfSnapshots(None, dataset=dataset)
            dl = DataLoader(ds, batch_size=self.batch_size, shuffle=False,
                            collate_fn=datasets.sequence_of_snapshots_fixed_n_collate, pin_memory=True)
        elif self.dataset_type == 'SequenceOfSnapshotsVariableN':
            ds = datasets.DatasetSequenceOfSnapshotsSortedNumSnapshots(None, dataset=dataset)
            dl = DataLoader(ds, batch_size=self.batch_size, shuffle=False,
                            collate_fn=datasets.sequence_of_snapshots_variable_n_collate, pin_memory=True)
        elif self.dataset_type == 'SequenceOfAntennas':
            ds = datasets.DatasetSequenceOfAntennas(None, dataset=dataset)
            dl = DataLoader(ds, batch_size=self.batch_size, shuffle=False,
                            collate_fn=datasets.sequence_of_antennas_collate, pin_memory=True)
        elif self.dataset_type == 'SequenceOfCovMatrixRows':
            ds = datasets.DatasetSequenceOfCovMatrixRows(None, dataset=dataset)
            dl = DataLoader(ds, batch_size=self.batch_size, shuffle=False,
                            collate_fn=datasets.sequence_of_cov_matrix_rows_collate, pin_memory=True)
        elif self.dataset_type == 'SequenceOfSnapshotsFixedNForSeq2Seq':
            ds = datasets.DatasetSequenceOfSnapshotsForSeq2Seq(None, self.trg_len, dataset=dataset)
            dl = DataLoader(ds, batch_size=self.batch_size, shuffle=False,
                            collate_fn=datasets.sequence_of_snapshots_fixed_n_for_seq2seq_collate, pin_memory=True)
        elif self.dataset_type == 'SequenceOfSnapshotsVariableNForSeq2Seq':
            ds = datasets.DatasetSequenceOfSnapshotsForSeq2SeqSortedNumSnapshots(None, self.trg_len, dataset=dataset)
            dl = DataLoader(ds, batch_size=self.batch_size, shuffle=False,
                            collate_fn=datasets.sequence_of_snapshots_variable_n_for_seq2seq_collate,
                            pin_memory=True)
        elif self.dataset_type == 'SequenceOfCovMatrixRowsForSeq2Seq':
            ds = datasets.DatasetSequenceOfCovMatrixRowsForSeq2Seq(None, self.trg_len, dataset=dataset)
            dl = DataLoader(ds, batch_size=self.batch_size, shuffle=False,
                            collate_fn=datasets.sequence_of_cov_matrix_rows_for_seq2seq_collate, pin_memory=True)
        else:
            dl = None
        return dl

    def predict_classifier(self, iterator):
        targets_tensor_list = []
        estimated_doas_list = []
        with torch.no_grad():
            for i, batch in tqdm(enumerate(iterator)):
                src = batch[0]
                trg = batch[1]

                src = src.to(self.device)
                trg = trg.to(self.device)

                output = self.model(src)

                output = output.to('cpu')
                trg = trg.to('cpu')

                output, estimated_doas = self.doa_est_func(output, trg)

                estimated_doas_list.extend(estimated_doas)

                targets_tensor_list.append(trg)

        targets_tensor = torch.vstack(targets_tensor_list)
        return estimated_doas_list, targets_tensor

    def predict_seq2seq_network(self, iterator):
        targets_tensor_list = []
        estimated_doas_list = []
        with torch.no_grad():
            for i, batch in tqdm(enumerate(iterator)):
                src = batch[0]
                trg = batch[1]

                src = src.to(self.device)
                trg = trg.to(self.device)

                output = self.model(src, trg, 0)

                output = output.to('cpu')
                trg = trg.to('cpu')

                output, estimated_doas, trg, prob = self.doa_est_func(output, trg)

                estimated_doas_list.extend(estimated_doas)

                targets_tensor_list.append(trg)

        targets_tensor = torch.vstack(targets_tensor_list)
        return estimated_doas_list, targets_tensor
