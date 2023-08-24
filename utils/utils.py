"""
This module contains various common methods.

"""

import math

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, \
    roc_curve


def plot_probs(probs_list, start_point, step=1, probs_fig_save_address=None, title=None, x_label=None, y_label=None):
    x_list = [x for x in range(start_point, start_point + step * len(probs_list), step)]
    plt.plot(x_list, probs_list)
    if title is not None:
        plt.title(title)
    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)
    plt.show()
    if probs_fig_save_address is not None:
        plt.savefig(probs_fig_save_address)


def plot_confusion_matrix(labels, preds, classes=None, names_of_classes=None, save_address=None):
    cm = confusion_matrix(labels, preds, labels=classes, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=names_of_classes)
    disp.plot()
    plt.title("confusion matrix")
    if save_address is not None:
        plt.savefig(save_address)
    plt.show()


def plot_roc_curve(true_y, y_prob, save_address=None):
    """
    plots the roc curve based of the probabilities
    """

    fpr, tpr, thresholds = roc_curve(true_y, y_prob)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("ROC curve")
    if save_address is not None:
        plt.savefig(save_address)
    plt.show()


def cal_auc(true_y, y_prob):
    auc = roc_auc_score(true_y, y_prob)
    return auc


def cal_eval_metrics(outputs, targets):
    p, r, f, s = precision_recall_fscore_support(targets, outputs, average='binary')
    acc = 0
    for i in range(len(targets)):
        if targets[i] == outputs[i]:
            acc += 1
    acc = acc / len(targets)

    return acc, r, p, f, s


def get_k_great_elements(input_list, k):
    out_list = sorted(input_list, key=lambda x: x[1], reverse=True)
    out_list = out_list[0:k]
    return out_list


def cal_doa_errors(estimated_doas_list, targets):
    doas_error_list = cal_doa_errors_for_samples(estimated_doas_list, targets)
    errors = torch.hstack(doas_error_list)
    return errors


def cal_doa_errors_for_samples(estimated_doas_list, targets):
    doas_error_list = []
    for i, trg in enumerate(targets):
        estimated_doas = estimated_doas_list[i]
        num_preds = len(estimated_doas)
        doas_true = torch.nonzero(trg).squeeze(dim=1)
        num_targets = len(doas_true)
        if num_preds == 0:
            doas_pred = doas_true[0] * torch.ones(size=doas_true.shape, dtype=torch.float)
        else:
            doas_pred = torch.FloatTensor(size=doas_true.shape)
            if num_preds > num_targets:
                estimated_doas = get_k_great_elements(estimated_doas, num_targets)
            for index, doa in enumerate(doas_true):
                doa_float = float(doa)
                distances_of_doa = [abs(doa_float - d) for d, a in estimated_doas]
                min_distance = min(distances_of_doa)
                min_index = distances_of_doa.index(min_distance)
                doas_pred[index] = estimated_doas[min_index][0]
        doas_error = doas_true - doas_pred
        doas_error_list.append(doas_error)
    return doas_error_list


def cal_mae(errors):
    n = errors.shape[0]
    sae = float(torch.norm(errors, p=1))
    mae = sae / n
    return mae


def cal_rmse(errors):
    n = errors.shape[0]
    rsse = float(torch.norm(errors, p=2))
    rmse = rsse / math.sqrt(n)
    return rmse


def find_local_maximums(input):
    input_length = int(input.shape[0])
    output = torch.zeros(size=input.shape, dtype=torch.float)
    maximums_index_list = []
    for i, element in enumerate(input):
        max_flag = False
        if i == 0:
            if element >= input[i + 1]:
                max_flag = True
        elif i == (input_length - 1):
            if element >= input[i - 1]:
                max_flag = True
        else:
            if element >= input[i + 1] and element >= input[i - 1]:
                max_flag = True
        if max_flag:
            maximums_index_list.append(i)
            output[i] = element
    return output, maximums_index_list


def estimate_doas_max_k(output, target):
    num_targets = torch.count_nonzero(target, dim=1)
    modified_output = torch.zeros(size=output.shape, dtype=torch.float)
    estimated_doas = []
    for i, out in enumerate(output):
        topk_out = torch.topk(out, int(num_targets[i]))
        topk_indices = topk_out.indices
        estimated_doas_i = []
        for index in topk_indices:
            modified_output[i, index] = 1
            estimated_doas_i.append((index, out[index]))
        estimated_doas.append(estimated_doas_i)
    return modified_output, estimated_doas


def estimate_doas_local_max_k(output, target):
    num_targets = torch.count_nonzero(target, dim=1)
    modified_output = torch.zeros(size=output.shape, dtype=torch.float)
    estimated_doas = []
    for i, out in enumerate(output):
        out_local_maximums, _ = find_local_maximums(out)
        topk_out = torch.topk(out_local_maximums, int(num_targets[i]))
        topk_indices = topk_out.indices
        estimated_doas_i = []
        for index in topk_indices:
            modified_output[i, index] = 1
            estimated_doas_i.append((index, out[index]))
        estimated_doas.append(estimated_doas_i)
    return modified_output, estimated_doas


def estimate_doas_local_max(output, threshold=1e-6):
    modified_output = torch.zeros(size=output.shape, dtype=torch.float)
    estimated_doas = []
    for i, out in enumerate(output):
        out_local_maximums, _ = find_local_maximums(out)
        topk_indices = torch.nonzero(out_local_maximums >= threshold).squeeze(dim=1)
        estimated_doas_i = []
        for index in topk_indices:
            modified_output[i, index] = 1
            estimated_doas_i.append((index, out[index]))
        estimated_doas.append(estimated_doas_i)
    return modified_output, estimated_doas


def estimate_doas_seq2seq(output, target):
    trg_len = target.shape[0]
    batch_size = target.shape[1]
    output_dim = output.shape[2] - 2
    output_indices = torch.argmax(output, dim=2)
    modified_probs, _ = torch.max(output, dim=0)
    modified_probs = modified_probs[:, 2:]
    estimated_doas = []
    modified_target = torch.zeros(size=(batch_size, output_dim), dtype=torch.float)
    modified_output = torch.zeros(size=(batch_size, output_dim), dtype=torch.float)
    for i in range(batch_size):
        trg = target[:, i]
        out = output_indices[:, i]
        estimated_doas_i = []
        for k in range(1, trg_len):
            if trg[k] == 1:
                break
            else:
                modified_target[i, trg[k] - 2] = 1
        for k in range(1, trg_len):
            if out[k] == 1:
                break
            elif out[k] == 0:
                continue
            else:
                modified_output[i, out[k] - 2] = 1
                estimated_doas_i.append((out[k] - 2, modified_probs[i, out[k] - 2]))
        estimated_doas.append(estimated_doas_i)
    return modified_output, estimated_doas, modified_target, modified_probs
