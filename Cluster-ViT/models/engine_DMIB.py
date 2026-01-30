"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable
import pandas as pd
import torch

import util.misc as utils
# from sksurv.metrics import integrated_brier_score, concordance_index_censored, concordance_index_ipcw
import numpy as np
from sksurv.linear_model.coxph import BreslowEstimator
from sksurv.metrics import concordance_index_censored
from torch.utils.tensorboard import SummaryWriter
# import matplotlib.pyplot as plt
# from sksurv.linear_model.coxph import BreslowEstimator
from torch.autograd import Variable



def mixup_data(sample, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        # lam = 1
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    (patientEmbedding, pos, keyPaddingMask, cluster, Dead, followUpTime, patientIdx) = sample
    batch_size = patientEmbedding.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    patientEmbedding = lam * patientEmbedding + (1 - lam) * patientEmbedding[index, :]
    pos = lam * pos + (1 - lam) * pos[index, :]
    if lam < 0.5:
        keyPaddingMask = keyPaddingMask[index, :]
        cluster = cluster[index, :]
    Dead_a, Dead_b = Dead, Dead[index]
    followUpTime_a, followUpTime_b = followUpTime, followUpTime[index]
    return patientEmbedding, pos, keyPaddingMask, cluster, Dead_a, Dead_b, followUpTime_a, followUpTime_b, lam


def mixup_criterion(criterion, outputs, followUpTime_a, followUpTime_b, Dead_a, Dead_b, lam):
    loss_dict_a = criterion(outputs, followUpTime_a, Dead_a)
    loss_dict_b = criterion(outputs, followUpTime_b, Dead_b)
    loss_dict = {k: lam * loss_dict_a[k] + (1 - lam) * loss_dict_b[k] for k in loss_dict_a}
    return loss_dict


def train_one_epoch_SAM(model: torch.nn.Module, criterion: torch.nn.Module,
                        data_loader: Iterable, optimizer: torch.optim.Optimizer,
                        device: torch.device, epoch: int, fold: int, tb_writer: SummaryWriter, max_norm: float = 0,
                        logSaveInterval=500, mixUp=True, SAM=True):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for batch_idx, sample in metric_logger.log_every(data_loader, print_freq, header):
        sample = [element.to(device) for element in sample]
        if mixUp:
            alpha = 2
            patientEmbedding, pos, keyPaddingMask, cluster, Dead_a, Dead_b, followUpTime_a, followUpTime_b, lam \
                = mixup_data(sample, alpha)
            if sum(Dead_a) == 0:
                metric_logger.update(loss=0)
                metric_logger.update(neg_likelihood=0)
                metric_logger.update(lr=optimizer.param_groups[0]["lr"])
                continue
            patientEmbedding, pos, keyPaddingMask, cluster, Dead_a, Dead_b, followUpTime_a, followUpTime_b = \
                map(Variable,
                    (patientEmbedding, pos, keyPaddingMask, cluster, Dead_a, Dead_b, followUpTime_a, followUpTime_b))
            outputs = model(patientEmbedding, pos, keyPaddingMask, cluster)
            loss_dict = mixup_criterion(criterion, outputs, followUpTime_a, followUpTime_b, Dead_a, Dead_b, lam)
        else:
            (patientEmbedding, pos, keyPaddingMask, cluster, Dead, followUpTime, patientIdx) = sample
            if sum(Dead) == 0:
                metric_logger.update(loss=0)
                metric_logger.update(neg_likelihood=0)
                metric_logger.update(lr=optimizer.param_groups[0]["lr"])
                continue
            outputs = model(patientEmbedding, pos, keyPaddingMask, cluster)
            loss_dict = criterion(outputs, followUpTime, Dead)

        weight_dict = criterion.weight_dict
        losses_1 = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)
        if SAM:
            optimizer.zero_grad()
            losses_1.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.first_step(zero_grad=True)

            outputs_2 = model(patientEmbedding, pos, keyPaddingMask, cluster)
            if mixUp:
                loss_dict_2 = mixup_criterion(criterion, outputs_2, followUpTime_a, followUpTime_b, Dead_a, Dead_b, lam)
            else:
                loss_dict_2 = criterion(outputs_2, followUpTime, Dead)
            losses_2 = sum(loss_dict_2[k] * weight_dict[k] for k in loss_dict_2.keys() if k in weight_dict)
            losses_2.backward()
            # second forward-backward pass
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.second_step(zero_grad=True)
        else:
            optimizer.zero_grad()
            losses_1.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        metric_logger.update(loss=loss_value)
        metric_logger.update(neg_likelihood=loss_dict_reduced['neg_likelihood'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_SAM_DMIB(model: torch.nn.Module, criterion: torch.nn.Module,
                             data_loader: Iterable, optimizer: torch.optim.Optimizer,
                             device: torch.device, epoch: int, fold: int, tb_writer: SummaryWriter, max_norm: float = 0,
                             logSaveInterval=500, mixUp=True, SAM=True):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    total_loss = []
    label_list = []
    score_list = []
    pred_list = []
    for batch_idx, sample in metric_logger.log_every(data_loader, print_freq, header):
        sample = [element.to(device) for element in sample]
        optimizer.zero_grad()
        if mixUp:
            alpha = 2
            patientEmbedding, pos, keyPaddingMask, cluster, Dead_a, Dead_b, followUpTime_a, followUpTime_b, lam \
                = mixup_data(sample, alpha)
            #
            # if sum(Dead_a) == 0:
            #     continue
            patientEmbedding, pos, keyPaddingMask, cluster, Dead_a, Dead_b, followUpTime_a, followUpTime_b = \
                map(Variable,
                    (patientEmbedding, pos, keyPaddingMask, cluster, Dead_a, Dead_b, followUpTime_a, followUpTime_b))
            outputs = model(patientEmbedding, pos, keyPaddingMask, cluster)
            try:
                loss_dict_a = criterion(outputs[0], Dead_a.float())
                loss_dict_b = criterion(outputs[0], Dead_b.float())
                loss_dict = lam * loss_dict_a + (1 - lam) * loss_dict_b
            except:
                loss_dict_a = criterion(torch.unsqueeze(outputs[0], 0), Dead_a.float())
                loss_dict_b = criterion(torch.unsqueeze(outputs[0], 0), Dead_b.float())
                loss_dict = lam * loss_dict_a + (1 - lam) * loss_dict_b

            loss_dict.backward()
            optimizer.step()
            total_loss.append(loss_dict)
            print('loss:', loss_dict)

            score = outputs[0].detach().cpu().numpy()
            probabilities = 1 / (1 + np.exp(-score))
            threshold = 0.5
            predictions = (probabilities >= threshold).astype(int)

            label_list.extend(Dead_b.cpu().numpy().tolist())

            try:
                score_list.extend(probabilities.tolist())
                pred_list.extend(predictions.tolist())
            except:
                score_list.extend(np.array([probabilities]).tolist())
                pred_list.extend(np.array([predictions]).tolist())

        else:
            (patientEmbedding, pos, keyPaddingMask, cluster, Dead, followUpTime, patientIdx) = sample
            # if sum(Dead) == 0:
            #     continue

            outputs = model(patientEmbedding, pos, keyPaddingMask, cluster)
            loss_dict = criterion(outputs[0], Dead.float())
            loss_dict.backward()
            print(loss_dict.shape)
            # loss_dict.backward()
            optimizer.step()
            total_loss.append(loss_dict)
            print('loss:', loss_dict)

            score = outputs[0].detach().cpu().numpy()
            probabilities = 1 / (1 + np.exp(-score))
            threshold = 0.5
            predictions = (probabilities >= threshold).astype(int)

            label_list.extend(Dead.cpu().numpy().tolist())

            try:
                score_list.extend(probabilities.tolist())
                pred_list.extend(predictions.tolist())
            except:
                score_list.extend(np.array([probabilities]).tolist())
                pred_list.extend(np.array([predictions]).tolist())


        metric_logger.update(loss=loss_dict)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    from sklearn.metrics import roc_auc_score
    try:
        auc_fuse_ob = roc_auc_score(label_list, score_list)
    except:
        auc_fuse_ob = 0
    acc_fuse_ob = accuracy_score(label_list, pred_list)
    cfm_fuse_ob = confusion_matrix(label_list, pred_list, labels=[0, 1])
    spec_fuse_ob = cfm_fuse_ob[0][0] / np.sum(cfm_fuse_ob[0])
    sens_fuse_ob = cfm_fuse_ob[1][1] / np.sum(cfm_fuse_ob[1])
    jw5 = sens_fuse_ob * 0.5 + spec_fuse_ob * (1 - 0.5)
    jw6 = sens_fuse_ob * 0.6 + spec_fuse_ob * (1 - 0.6)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    return torch.stack(total_loss).mean().item(), auc_fuse_ob, acc_fuse_ob, spec_fuse_ob, sens_fuse_ob, jw5, jw6


@torch.no_grad()
def evaluate_DMIB(model, criterion, data_loader, device, output_dir, status, fold = 0):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = status + ' evaluate:'
    event_indicator = np.asarray([], dtype=bool)
    event_time = np.asarray([])
    estimate = np.asarray([])
    A = []
    print_freq = 10
    total_loss = []
    label_list = []
    score_list = []
    pred_list = []
    patient_name = []
    outputs_list = []
    for batch_id, sample in metric_logger.log_every(data_loader, print_freq, header):
        sample = [element.to(device) for element in sample]
        (patientEmbedding, pos, keyPaddingMask, cluster, Dead, followUpTime, patientIdx) = sample
        # if sum(Dead) == 0:
        #     continue
        event_indicator = np.append(event_indicator, Dead.cpu().numpy().astype(bool))
        event_time = np.append(event_time, followUpTime.cpu().numpy())

        outputs = model(patientEmbedding, pos, keyPaddingMask, cluster)
        estimate = np.append(estimate, outputs[0].cpu().numpy())
        # A.append(outputs[2].cpu().numpy())
        try:
            loss_dict = criterion(outputs[0], Dead.float())
        except:
            print(outputs[0].shape)
            loss_dict = criterion(torch.unsqueeze(outputs[0], 0), Dead.float())
        total_loss.append(loss_dict)
        score = outputs[0].detach().cpu().numpy()
        probabilities = 1 / (1 + np.exp(-score))
        threshold = 0.5
        predictions = (probabilities >= threshold).astype(int)

        label_list.extend(Dead.cpu().numpy().tolist())
        patient_name.extend(patientIdx.cpu().numpy().tolist())

        try:
            outputs_list.extend(outputs[0].cpu().numpy().tolist())
            score_list.extend(probabilities.tolist())
            pred_list.extend(predictions.tolist())
        except:
            outputs_list.append(outputs[0].cpu().numpy())
            score_list.append(probabilities)
            pred_list.append(predictions)





        metric_logger.update(loss=loss_dict.item())
    df_recording = pd.DataFrame()
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score
    from sklearn.metrics import roc_auc_score
    try:
        auc_fuse_ob = roc_auc_score(label_list, score_list)
    except:
        auc_fuse_ob = 0
    acc_fuse_ob = accuracy_score(label_list, pred_list)
    cfm_fuse_ob = confusion_matrix(label_list, pred_list, labels=[0, 1])
    spec_fuse_ob = cfm_fuse_ob[0][0] / np.sum(cfm_fuse_ob[0])
    sens_fuse_ob = cfm_fuse_ob[1][1] / np.sum(cfm_fuse_ob[1])
    jw5 = sens_fuse_ob * 0.5 + spec_fuse_ob * (1 - 0.5)
    jw6 = sens_fuse_ob * 0.6 + spec_fuse_ob * (1 - 0.6)
    precision_binary = precision_score(label_list, pred_list, average='binary')
    f1 = f1_score(label_list, pred_list, average='macro')

    return torch.stack(total_loss).mean().item(), auc_fuse_ob, acc_fuse_ob, spec_fuse_ob, sens_fuse_ob, jw5, jw6

@torch.no_grad()
def test(model, criterion, test_data_loader, train_data_loader, device, output_dir, coxBiomarkerRisk=None):
    model.eval()
    criterion.eval()

    event_indicator_train = np.asarray([], dtype=bool)
    event_time_train = np.asarray([])
    estimate_train = np.asarray([])
    event_indicator_test = np.asarray([], dtype=bool)
    event_time_test = np.asarray([])
    estimate_test = np.asarray([])
    survival_score_patches = np.asarray([])
    all_clusters = np.asarray([])
    print_freq = 10
    max_num_cluster = model.max_num_cluster
    meanClusterRisk = []
    stdClusterRisk = []
    meanClusterRisk_test = []
    stdClusterRisk_test = []
    survival_score_patches_test = np.asarray([])
    all_clusters_test = np.asarray([])
    all_pos = np.empty((0, 3), dtype=np.float32)
    all_patientIdx = np.asarray([])
    all_pos_test = np.empty((0, 3), dtype=np.float32)
    all_patientIdx_test = np.asarray([])
    all_mean_survival_score = []
    all_percentage = []
    all_patient_score = []
    all_label = []
    all_predictions = []
    all_patchidx = np.asarray([])
    all_patchidx_test = np.asarray([])
    LearnedClusterRisk = model.survival_score_cluster[0:max_num_cluster].cpu().tolist()

    for batch_id, sample in enumerate(train_data_loader, print_freq):
        # samples = samples.to(device)
        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        sample = [element.to(device) for element in sample]
        (patientEmbedding, pos, keyPaddingMask, cluster, Dead, followUpTime, patientIdx) = sample
        event_indicator_train = np.append(event_indicator_train, Dead.cpu().numpy().astype(bool))
        event_time_train = np.append(event_time_train, followUpTime.cpu().numpy())
        outputs = model(patientEmbedding, pos, keyPaddingMask, cluster)
        estimate_train = np.append(estimate_train, outputs[0].cpu().numpy())
        survival_score_patches = np.append(survival_score_patches, outputs[1].cpu().numpy().flatten())
        all_clusters = np.append(all_clusters, cluster.cpu().numpy().flatten())
        all_pos = np.vstack((all_pos, pos.cpu().numpy().reshape(-1, 3)))
        all_patientIdx = np.append(all_patientIdx, patientIdx.cpu().numpy().flatten())
        # all_patchidx = np.append(all_patientIdx, patchidx.cpu().numpy().flatten())
        if np.mean(survival_score_patches[all_clusters == 1])>-1000:
            pass
        else:
            print('error')


    for batch_id, sample in enumerate(test_data_loader, print_freq):
        sample = [element.to(device) for element in sample]
        (patientEmbedding, pos, keyPaddingMask, cluster, Dead, followUpTime, patientIdx) = sample
        event_indicator_test = np.append(event_indicator_test, Dead.cpu().numpy().astype(bool))
        event_time_test = np.append(event_time_test, followUpTime.cpu().numpy())
        outputs = model(patientEmbedding, pos, keyPaddingMask, cluster)
        estimate_test = np.append(estimate_test, outputs[0].cpu().numpy())
        survival_score_patches_test = np.append(survival_score_patches_test, outputs[1].cpu().numpy().flatten())
        all_clusters_test = np.append(all_clusters_test, cluster.cpu().numpy().flatten())
        all_pos_test = np.vstack((all_pos_test, pos.cpu().numpy().reshape(-1, 3)))
        all_patientIdx_test = np.append(all_patientIdx_test, patientIdx.cpu().numpy().flatten())
        # all_patchidx_test = np.append(all_patientIdx_test, patchidx.cpu().numpy().flatten())

        probabilities = 1 / (1 + np.exp(-outputs[0].cpu().numpy()))
        threshold = 0.5
        predictions = (probabilities >= threshold).astype(int)
        all_predictions.extend(predictions)

        all_patient_score.extend(outputs[0].cpu().numpy())
        all_risk = outputs[3].cpu().numpy()
        all_mean_survival_score.extend(outputs[4].cpu().numpy())
        all_percentage.extend(outputs[5].cpu().numpy())
        all_label.extend(Dead.cpu().numpy())

    for i in range(0, max_num_cluster):
        meanClusterRisk.append(np.mean(survival_score_patches[all_clusters == i]))
        stdClusterRisk.append(np.std(survival_score_patches[all_clusters == i]))
        meanClusterRisk_test.append(np.mean(survival_score_patches_test[all_clusters_test == i]))
        stdClusterRisk_test.append(np.std(survival_score_patches_test[all_clusters_test == i]))
    clusterRiskDataFrame = pd.DataFrame({'meanClusterRisk': meanClusterRisk, 'stdClusterRisk': stdClusterRisk,
                                         'LearnedClusterRisk': LearnedClusterRisk})
    clusterRiskDataFrame.to_csv('/media/NAS04/zihao/PrognosticBiomarkerDiscovery/Cluster-ViT/experiments_fld_vit/DMIB_bs2_mixupa_skip00_update_1230/clusterRisk_fold.csv')
    clusterRiskDataFrame_test = pd.DataFrame(
        {'meanClusterRisk': meanClusterRisk_test, 'stdClusterRisk': stdClusterRisk_test,
            'LearnedClusterRisk': LearnedClusterRisk})
    clusterRiskDataFrame_test.to_csv('/media/NAS04/zihao/PrognosticBiomarkerDiscovery/Cluster-ViT/experiments_fld_vit/DMIB_bs2_mixupa_skip00_update_1230/clusterRisk_test_fold.csv')


def save_final(result_bestloss_train_folds, result_bestloss_val_folds,result_bestloss_test_folds,
               result_bestauc_train_folds, result_bestauc_val_folds, result_bestauc_test_folds,
               result_bestjw5_train_folds,  result_bestjw5_val_folds, result_bestjw5_test_folds,
               result_bestjw6_train_folds,  result_bestjw6_val_folds, result_bestjw6_test_folds, save_result):

    result_bestloss_train_folds = np.array(result_bestloss_train_folds)
    result_bestloss_val_folds = np.array(result_bestloss_val_folds)
    result_bestloss_test_folds = np.array(result_bestloss_test_folds)
    result_bestauc_train_folds = np.array(result_bestauc_train_folds)
    result_bestauc_val_folds = np.array(result_bestauc_val_folds)
    result_bestauc_test_folds = np.array(result_bestauc_test_folds)
    result_bestjw5_train_folds = np.array(result_bestjw5_train_folds)
    result_bestjw5_val_folds = np.array(result_bestjw5_val_folds)
    result_bestjw5_test_folds = np.array(result_bestjw5_test_folds)
    result_bestjw6_train_folds = np.array(result_bestjw6_train_folds)
    result_bestjw6_val_folds = np.array(result_bestjw6_val_folds)
    result_bestjw6_test_folds = np.array(result_bestjw6_test_folds)

    ##### selecting model: best loss #######
    Bestloss_Train_Auc = "{:.2f} + {:.2f}".format(result_bestloss_train_folds[:, 1].mean(), result_bestloss_train_folds[:, 1].std())
    Bestloss_Train_Acc = "{:.2f} + {:.2f}".format(result_bestloss_train_folds[:, 2].mean(), result_bestloss_train_folds[:, 2].std())
    Bestloss_Train_Sens = "{:.2f} + {:.2f}".format(result_bestloss_train_folds[:, 3].mean(), result_bestloss_train_folds[:, 3].std())
    Bestloss_Train_Spec = "{:.2f} + {:.2f}".format(result_bestloss_train_folds[:, 4].mean(), result_bestloss_train_folds[:, 4].std())
    Bestloss_Train_jw5 = "{:.2f} + {:.2f}".format(result_bestloss_train_folds[:, 5].mean(), result_bestloss_train_folds[:, 5].std())
    Bestloss_Train_jw6 = "{:.2f} + {:.2f}".format(result_bestloss_train_folds[:, 6].mean(), result_bestloss_train_folds[:, 6].std())

    Bestloss_Val_Auc = "{:.2f} + {:.2f}".format(result_bestloss_val_folds[:, 1].mean(), result_bestloss_val_folds[:, 1].std())
    Bestloss_Val_Acc = "{:.2f} + {:.2f}".format(result_bestloss_val_folds[:, 2].mean(), result_bestloss_val_folds[:, 2].std())
    Bestloss_Val_Sens = "{:.2f} + {:.2f}".format(result_bestloss_val_folds[:, 3].mean(), result_bestloss_val_folds[:, 3].std())
    Bestloss_Val_Spec = "{:.2f} + {:.2f}".format(result_bestloss_val_folds[:, 4].mean(), result_bestloss_val_folds[:, 4].std())
    Bestloss_Val_jw5 = "{:.2f} + {:.2f}".format(result_bestloss_val_folds[:, 5].mean(), result_bestloss_val_folds[:, 5].std())
    Bestloss_Val_jw6 = "{:.2f} + {:.2f}".format(result_bestloss_val_folds[:, 6].mean(), result_bestloss_val_folds[:, 6].std())

    Bestloss_Test_Auc = "{:.2f} + {:.2f}".format(result_bestloss_test_folds[:, 1].mean(), result_bestloss_test_folds[:, 1].std())
    Bestloss_Test_Acc = "{:.2f} + {:.2f}".format(result_bestloss_test_folds[:, 2].mean(), result_bestloss_test_folds[:, 2].std())
    Bestloss_Test_Sens = "{:.2f} + {:.2f}".format(result_bestloss_test_folds[:, 3].mean(), result_bestloss_test_folds[:, 3].std())
    Bestloss_Test_Spec = "{:.2f} + {:.2f}".format(result_bestloss_test_folds[:, 4].mean(), result_bestloss_test_folds[:, 4].std())
    Bestloss_Test_jw5 = "{:.2f} + {:.2f}".format(result_bestloss_test_folds[:, 5].mean(), result_bestloss_test_folds[:, 5].std())
    Bestloss_Test_jw6 = "{:.2f} + {:.2f}".format(result_bestloss_test_folds[:, 6].mean(), result_bestloss_test_folds[:, 6].std())

    Bestloss_Test_index = np.where(result_bestloss_val_folds[:, 1] == result_bestloss_val_folds[:, 1].max())[0][0]
    Bestloss_Test_Auc_bestepoch = "{:.2f}".format(result_bestloss_test_folds[Bestloss_Test_index, 1])
    Bestloss_Test_Acc_bestepoch = "{:.2f}".format(result_bestloss_test_folds[Bestloss_Test_index, 2])
    Bestloss_Test_Sens_bestepoch = "{:.2f}".format(result_bestloss_test_folds[Bestloss_Test_index, 3])
    Bestloss_Test_Spec_bestepoch = "{:.2f}".format(result_bestloss_test_folds[Bestloss_Test_index, 4])
    Bestloss_Test_jw5_bestepoch = "{:.2f}".format(result_bestloss_test_folds[Bestloss_Test_index, 5])
    Bestloss_Test_jw6_bestepoch = "{:.2f}".format(result_bestloss_test_folds[Bestloss_Test_index, 6])

    ##### selecting model: best auc #######
    Bestauc_Train_Auc = "{:.2f} + {:.2f}".format(result_bestauc_train_folds[:, 1].mean(), result_bestauc_train_folds[:, 1].std())
    Bestauc_Train_Acc = "{:.2f} + {:.2f}".format(result_bestauc_train_folds[:, 2].mean(), result_bestauc_train_folds[:, 2].std())
    Bestauc_Train_Sens = "{:.2f} + {:.2f}".format(result_bestauc_train_folds[:, 3].mean(), result_bestauc_train_folds[:, 3].std())
    Bestauc_Train_Spec = "{:.2f} + {:.2f}".format(result_bestauc_train_folds[:, 4].mean(), result_bestauc_train_folds[:, 4].std())
    Bestauc_Train_jw5 = "{:.2f} + {:.2f}".format(result_bestauc_train_folds[:, 5].mean(), result_bestauc_train_folds[:, 5].std())
    Bestauc_Train_jw6 = "{:.2f} + {:.2f}".format(result_bestauc_train_folds[:, 6].mean(), result_bestauc_train_folds[:, 6].std())

    Bestauc_Val_Auc = "{:.2f} + {:.2f}".format(result_bestauc_val_folds[:, 1].mean(), result_bestauc_val_folds[:, 1].std())
    Bestauc_Val_Acc = "{:.2f} + {:.2f}".format(result_bestauc_val_folds[:, 2].mean(), result_bestauc_val_folds[:, 2].std())
    Bestauc_Val_Sens = "{:.2f} + {:.2f}".format(result_bestauc_val_folds[:, 3].mean(), result_bestauc_val_folds[:, 3].std())
    Bestauc_Val_Spec = "{:.2f} + {:.2f}".format(result_bestauc_val_folds[:, 4].mean(), result_bestauc_val_folds[:, 4].std())
    Bestauc_Val_jw5 = "{:.2f} + {:.2f}".format(result_bestauc_val_folds[:, 5].mean(), result_bestauc_val_folds[:,5].std())
    Bestauc_Val_jw6 = "{:.2f} + {:.2f}".format(result_bestauc_val_folds[:, 6].mean(), result_bestauc_val_folds[:, 6].std())

    Bestauc_Test_Auc = "{:.2f} + {:.2f}".format(result_bestauc_test_folds[:, 1].mean(), result_bestauc_test_folds[:, 1].std())
    Bestauc_Test_Acc = "{:.2f} + {:.2f}".format(result_bestauc_test_folds[:, 2].mean(), result_bestauc_test_folds[:, 2].std())
    Bestauc_Test_Sens = "{:.2f} + {:.2f}".format(result_bestauc_test_folds[:, 3].mean(), result_bestauc_test_folds[:, 3].std())
    Bestauc_Test_Spec = "{:.2f} + {:.2f}".format(result_bestauc_test_folds[:, 4].mean(), result_bestauc_test_folds[:, 4].std())
    Bestauc_Test_jw5 = "{:.2f} + {:.2f}".format(result_bestauc_test_folds[:, 5].mean(), result_bestauc_test_folds[:, 5].std())
    Bestauc_Test_jw6 = "{:.2f} + {:.2f}".format(result_bestauc_test_folds[:, 6].mean(), result_bestauc_test_folds[:, 6].std())

    Bestauc_Test_index = np.where(result_bestauc_val_folds[:, 1] == result_bestauc_val_folds[:, 1].max())[0][0]
    Bestauc_Test_Auc_bestepoch = "{:.2f}".format(result_bestauc_test_folds[Bestauc_Test_index, 1])
    Bestauc_Test_Acc_bestepoch = "{:.2f}".format(result_bestauc_test_folds[Bestauc_Test_index, 2])
    Bestauc_Test_Sens_bestepoch = "{:.2f}".format(result_bestauc_test_folds[Bestauc_Test_index, 3])
    Bestauc_Test_Spec_bestepoch = "{:.2f}".format(result_bestauc_test_folds[Bestauc_Test_index, 4])
    Bestauc_Test_jw5_bestepoch = "{:.2f}".format(result_bestauc_test_folds[Bestauc_Test_index, 5])
    Bestauc_Test_jw6_bestepoch = "{:.2f}".format(result_bestauc_test_folds[Bestauc_Test_index, 6])

    ##### selecting model: best jw #######
    Bestjw5_Train_Auc = "{:.2f} + {:.2f}".format(result_bestjw5_train_folds[:, 1].mean(), result_bestjw5_train_folds[:, 1].std())
    Bestjw5_Train_Acc = "{:.2f} + {:.2f}".format(result_bestjw5_train_folds[:, 2].mean(), result_bestjw5_train_folds[:, 2].std())
    Bestjw5_Train_Sens = "{:.2f} + {:.2f}".format(result_bestjw5_train_folds[:, 3].mean(), result_bestjw5_train_folds[:, 3].std())
    Bestjw5_Train_Spec = "{:.2f} + {:.2f}".format(result_bestjw5_train_folds[:, 4].mean(), result_bestjw5_train_folds[:, 4].std())
    Bestjw5_Train_jw5 = "{:.2f} + {:.2f}".format(result_bestjw5_train_folds[:, 5].mean(), result_bestjw5_train_folds[:, 5].std())
    Bestjw5_Train_jw6 = "{:.2f} + {:.2f}".format(result_bestjw5_train_folds[:, 6].mean(), result_bestjw5_train_folds[:, 6].std())

    Bestjw5_Val_Auc = "{:.2f} + {:.2f}".format(result_bestjw5_val_folds[:, 1].mean(), result_bestjw5_val_folds[:, 1].std())
    Bestjw5_Val_Acc = "{:.2f} + {:.2f}".format(result_bestjw5_val_folds[:, 2].mean(), result_bestjw5_val_folds[:, 2].std())
    Bestjw5_Val_Sens = "{:.2f} + {:.2f}".format(result_bestjw5_val_folds[:, 3].mean(), result_bestjw5_val_folds[:, 3].std())
    Bestjw5_Val_Spec = "{:.2f} + {:.2f}".format(result_bestjw5_val_folds[:, 4].mean(), result_bestjw5_val_folds[:, 4].std())
    Bestjw5_Val_jw5 = "{:.2f} + {:.2f}".format(result_bestjw5_val_folds[:, 5].mean(), result_bestjw5_val_folds[:, 5].std())
    Bestjw5_Val_jw6 = "{:.2f} + {:.2f}".format(result_bestjw5_val_folds[:, 6].mean(), result_bestjw5_val_folds[:, 6].std())

    Bestjw5_Test_Auc = "{:.2f} + {:.2f}".format(result_bestjw5_test_folds[:, 1].mean(), result_bestjw5_test_folds[:, 1].std())
    Bestjw5_Test_Acc = "{:.2f} + {:.2f}".format(result_bestjw5_test_folds[:, 2].mean(), result_bestjw5_test_folds[:, 2].std())
    Bestjw5_Test_Sens = "{:.2f} + {:.2f}".format(result_bestjw5_test_folds[:, 3].mean(), result_bestjw5_test_folds[:, 3].std())
    Bestjw5_Test_Spec = "{:.2f} + {:.2f}".format(result_bestjw5_test_folds[:, 4].mean(), result_bestjw5_test_folds[:, 4].std())
    Bestjw5_Test_jw5 = "{:.2f} + {:.2f}".format(result_bestjw5_test_folds[:, 5].mean(), result_bestjw5_test_folds[:, 5].std())
    Bestjw5_Test_jw6 = "{:.2f} + {:.2f}".format(result_bestjw5_test_folds[:, 6].mean(), result_bestjw5_test_folds[:, 6].std())

    Bestjw5_Test_index = np.where(result_bestjw5_val_folds[:, 5] == result_bestjw5_val_folds[:, 5].max())[0][0]
    Bestjw5_Test_Auc_bestepoch = "{:.2f}".format(result_bestjw5_test_folds[Bestjw5_Test_index, 1])
    Bestjw5_Test_Acc_bestepoch = "{:.2f}".format(result_bestjw5_test_folds[Bestjw5_Test_index, 2])
    Bestjw5_Test_Sens_bestepoch = "{:.2f}".format(result_bestjw5_test_folds[Bestjw5_Test_index, 3])
    Bestjw5_Test_Spec_bestepoch = "{:.2f}".format(result_bestjw5_test_folds[Bestjw5_Test_index, 4])
    Bestjw5_Test_jw5_bestepoch = "{:.2f}".format(result_bestjw5_test_folds[Bestjw5_Test_index, 5])
    Bestjw5_Test_jw6_bestepoch = "{:.2f}".format(result_bestjw5_test_folds[Bestjw5_Test_index, 6])

    Bestjw6_Train_Auc = "{:.2f} + {:.2f}".format(result_bestjw6_train_folds[:, 1].mean(), result_bestjw6_train_folds[:, 1].std())
    Bestjw6_Train_Acc = "{:.2f} + {:.2f}".format(result_bestjw6_train_folds[:, 2].mean(), result_bestjw6_train_folds[:, 2].std())
    Bestjw6_Train_Sens = "{:.2f} + {:.2f}".format(result_bestjw6_train_folds[:, 3].mean(), result_bestjw6_train_folds[:, 3].std())
    Bestjw6_Train_Spec = "{:.2f} + {:.2f}".format(result_bestjw6_train_folds[:, 4].mean(), result_bestjw6_train_folds[:, 4].std())
    Bestjw6_Train_jw5 = "{:.2f} + {:.2f}".format(result_bestjw6_train_folds[:, 5].mean(), result_bestjw6_train_folds[:, 5].std())
    Bestjw6_Train_jw6 = "{:.2f} + {:.2f}".format(result_bestjw6_train_folds[:, 6].mean(), result_bestjw6_train_folds[:, 6].std())

    Bestjw6_Val_Auc = "{:.2f} + {:.2f}".format(result_bestjw6_val_folds[:, 1].mean(), result_bestjw6_val_folds[:, 1].std())
    Bestjw6_Val_Acc = "{:.2f} + {:.2f}".format(result_bestjw6_val_folds[:, 2].mean(), result_bestjw6_val_folds[:, 2].std())
    Bestjw6_Val_Sens = "{:.2f} + {:.2f}".format(result_bestjw6_val_folds[:, 3].mean(), result_bestjw6_val_folds[:, 3].std())
    Bestjw6_Val_Spec = "{:.2f} + {:.2f}".format(result_bestjw6_val_folds[:, 4].mean(), result_bestjw6_val_folds[:, 4].std())
    Bestjw6_Val_jw5 = "{:.2f} + {:.2f}".format(result_bestjw6_val_folds[:, 3].mean(), result_bestjw6_val_folds[:, 5].std())
    Bestjw6_Val_jw6 = "{:.2f} + {:.2f}".format(result_bestjw6_val_folds[:, 4].mean(), result_bestjw6_val_folds[:, 6].std())

    Bestjw6_Test_Auc = "{:.2f} + {:.2f}".format(result_bestjw6_test_folds[:, 1].mean(), result_bestjw6_test_folds[:, 1].std())
    Bestjw6_Test_Acc = "{:.2f} + {:.2f}".format(result_bestjw6_test_folds[:, 2].mean(), result_bestjw6_test_folds[:, 2].std())
    Bestjw6_Test_Sens = "{:.2f} + {:.2f}".format(result_bestjw6_test_folds[:, 3].mean(), result_bestjw6_test_folds[:, 3].std())
    Bestjw6_Test_Spec = "{:.2f} + {:.2f}".format(result_bestjw6_test_folds[:, 4].mean(), result_bestjw6_test_folds[:, 4].std())
    Bestjw6_Test_jw5 = "{:.2f} + {:.2f}".format(result_bestjw6_test_folds[:, 5].mean(), result_bestjw6_test_folds[:, 5].std())
    Bestjw6_Test_jw6 = "{:.2f} + {:.2f}".format(result_bestjw6_test_folds[:, 6].mean(), result_bestjw6_test_folds[:, 6].std())

    Bestjw6_Test_index = np.where(result_bestjw6_val_folds[:, 6] == result_bestjw6_val_folds[:, 6].max())[0][0]
    Bestjw6_Test_Auc_bestepoch = "{:.2f}".format(result_bestjw6_test_folds[Bestjw6_Test_index, 1])
    Bestjw6_Test_Acc_bestepoch = "{:.2f}".format(result_bestjw6_test_folds[Bestjw6_Test_index, 2])
    Bestjw6_Test_Sens_bestepoch = "{:.2f}".format(result_bestjw6_test_folds[Bestjw6_Test_index, 3])
    Bestjw6_Test_Spec_bestepoch = "{:.2f}".format(result_bestjw6_test_folds[Bestjw6_Test_index, 4])
    Bestjw6_Test_jw5_bestepoch = "{:.2f}".format(result_bestjw6_test_folds[Bestjw6_Test_index, 5])
    Bestjw6_Test_jw6_bestepoch = "{:.2f}".format(result_bestjw6_test_folds[Bestjw6_Test_index, 6])

    ## print
    print("Model selected by the best auc")
    print("Fold 1-5-Bestauc (Train)\t  AUC: {},\t ACC: {},\t Sens: {},\t Spec: {},\t jw5: {},\t jw6:{}".format(Bestauc_Train_Auc,
                                                                                          Bestauc_Train_Acc,
                                                                                          Bestauc_Train_Sens,
                                                                                          Bestauc_Train_Spec,Bestauc_Train_jw5,Bestauc_Train_jw6))
    print("Fold 1-5-Bestauc (Val)\t\t  AUC: {},\t ACC: {},\t Sens: {},\t Spec: {},\t jw5: {},\t jw6:{}".format(Bestauc_Val_Auc, Bestauc_Val_Acc,
                                                                                          Bestauc_Val_Sens,
                                                                                          Bestauc_Val_Spec, Bestauc_Val_jw5, Bestauc_Val_jw6))
    print("Fold 1-5-Bestauc (Test)\t\t AUC: {},\t ACC: {},\t Sens: {},\t Spec: {},\t jw5: {},\t jw6:{}".format(Bestauc_Test_Auc, Bestauc_Test_Acc,
                                                                                        Bestauc_Test_Sens,
                                                                                        Bestauc_Test_Spec, Bestauc_Test_jw5, Bestauc_Test_jw6))
    print("Fold 1-5-Bestauc (Test*)\t\t AUC: {},\t ACC: {},\t Sens: {},\t Spec: {},\t jw5: {},\t jw6:{}".format(Bestauc_Test_Auc_bestepoch, Bestauc_Test_Acc_bestepoch,
                                                                                        Bestauc_Test_Sens_bestepoch,
                                                                                        Bestauc_Test_Spec_bestepoch, Bestauc_Test_jw5_bestepoch, Bestauc_Test_jw6_bestepoch))

    print("Model selected by the best loss")
    print("Fold 1-5-Bestloss (Train)\t  AUC: {},\t ACC: {},\t Sens: {},\t Spec: {},\t jw5: {},\t jw6:{}".format(Bestloss_Train_Auc,
                                                                                           Bestloss_Train_Acc,
                                                                                           Bestloss_Train_Sens,
                                                                                           Bestloss_Train_Spec, Bestloss_Train_jw5, Bestauc_Test_jw6))
    print("Fold 1-5-Bestloss (Val)\t\t  AUC: {},\t ACC: {},\t Sens: {},\t Spec: {},\t jw5: {},\t jw6:{}".format(Bestloss_Val_Auc, Bestloss_Val_Acc,
                                                                                         Bestloss_Val_Sens,
                                                                                         Bestloss_Val_Spec, Bestloss_Val_jw5, Bestloss_Val_jw6))
    print("Fold 1-5-Bestloss (Test)\t\t AUC: {},\t ACC: {},\t Sens: {},\t Spec: {},\t jw5: {},\t jw6:{}".format(Bestloss_Test_Auc,
                                                                                           Bestloss_Test_Acc,
                                                                                           Bestloss_Test_Sens,
                                                                                           Bestloss_Test_Spec, Bestloss_Test_jw5, Bestloss_Test_jw6))
    print("Fold 1-5-Bestloss (Test*)\t\t AUC: {},\t ACC: {},\t Sens: {},\t Spec: {},\t jw5: {},\t jw6:{}".format(Bestloss_Test_Auc_bestepoch,
                                                                                           Bestloss_Test_Acc_bestepoch,
                                                                                           Bestloss_Test_Sens_bestepoch,
                                                                                           Bestloss_Test_Spec_bestepoch, Bestloss_Test_jw5_bestepoch, Bestloss_Test_jw6_bestepoch))

    print("Model selected by the best jw0.5")
    print("Fold 1-5-Bestjw5 (Train)\t  AUC: {},\t ACC: {},\t Sens: {},\t Spec: {},\t jw5: {},\t jw6:{}".format(Bestjw5_Train_Auc,
                                                                                           Bestjw5_Train_Acc,
                                                                                           Bestjw5_Train_Sens,
                                                                                           Bestjw5_Train_Spec,Bestjw5_Train_jw5,
                                                                                           Bestjw5_Train_jw6))
    print("Fold 1-5-Bestjw5 (Val)\t\t  AUC: {},\t ACC: {},\t Sens: {},\t Spec: {},\t jw5: {},\t jw6:{}".format(Bestjw5_Val_Auc, Bestjw5_Val_Acc,
                                                                                         Bestjw5_Val_Sens,
                                                                                         Bestjw5_Val_Spec,Bestjw5_Val_jw5,
                                                                                         Bestjw5_Val_jw6))
    print("Fold 1-5-Bestjw5 (Test*)\t\t AUC: {},\t ACC: {},\t Sens: {},\t Spec: {},\t jw5: {},\t jw6:{}".format(Bestjw5_Test_Auc_bestepoch,
                                                                                           Bestjw5_Test_Acc_bestepoch,
                                                                                           Bestjw5_Test_Sens_bestepoch,
                                                                                           Bestjw5_Test_Spec_bestepoch,Bestjw5_Test_jw5_bestepoch,
                                                                                           Bestjw5_Test_jw6_bestepoch))

    print("Model selected by the best jw0.6")
    print("Fold 1-5-Bestjw6 (Train)\t  AUC: {},\t ACC: {},\t Sens: {},\t Spec: {},\t jw5: {},\t jw6:{}".format(Bestjw6_Train_Auc,
                                                                                           Bestjw6_Train_Acc,
                                                                                           Bestjw6_Train_Sens,
                                                                                           Bestjw6_Train_Spec,Bestjw6_Train_jw5,
                                                                                           Bestjw6_Train_jw6))
    print("Fold 1-5-Bestjw6 (Val)\t\t  AUC: {},\t ACC: {},\t Sens: {},\t Spec: {},\t jw5: {},\t jw6:{}".format(Bestjw6_Val_Auc, Bestjw6_Val_Acc,
                                                                                         Bestjw6_Val_Sens,
                                                                                         Bestjw6_Val_Spec,Bestjw6_Val_jw5,
                                                                                         Bestjw6_Val_jw6))
    print("Fold 1-5-Bestjw6 (Test)\t\t AUC: {},\t ACC: {},\t Sens: {},\t Spec: {},\t jw5: {},\t jw6:{}".format(Bestjw6_Test_Auc,
                                                                                           Bestjw6_Test_Acc,
                                                                                           Bestjw6_Test_Sens,
                                                                                           Bestjw6_Test_Spec,Bestjw6_Test_jw5,
                                                                                           Bestjw6_Test_jw6))
    print("Fold 1-5-Bestjw6 (Test*)\t\t AUC: {},\t ACC: {},\t Sens: {},\t Spec: {},\t jw5: {},\t jw6:{}".format(Bestjw6_Test_Auc_bestepoch,
                                                                                           Bestjw6_Test_Acc_bestepoch,
                                                                                           Bestjw6_Test_Sens_bestepoch,
                                                                                           Bestjw6_Test_Spec_bestepoch,Bestjw6_Test_jw5_bestepoch,
                                                                                           Bestjw6_Test_jw6_bestepoch))

    ## save as csv
    result = pd.DataFrame(columns=('Stage', 'AUC', 'ACC', 'Sens', 'Spec'))

    result = result.append(pd.DataFrame(
        {'Stage': 'Bestloss_Train', 'AUC': [Bestloss_Train_Auc], 'ACC': [Bestloss_Train_Acc], 'Sens': [Bestloss_Train_Sens], 'Spec': [Bestloss_Train_Spec], 'JW5': [Bestloss_Train_jw5], 'JW6': [Bestloss_Train_jw6]}), ignore_index=True)
    result = result.append(pd.DataFrame(
        {'Stage': 'Bestloss_Val', 'AUC': [Bestloss_Val_Auc], 'ACC': [Bestloss_Val_Acc], 'Sens': [Bestloss_Val_Sens], 'Spec': [Bestloss_Val_Spec], 'JW5': [Bestloss_Val_jw5], 'JW6': [Bestloss_Val_jw6]}), ignore_index=True)
    result = result.append(pd.DataFrame(
        {'Stage': 'Bestloss_Test', 'AUC': [Bestloss_Test_Auc], 'ACC': [Bestloss_Test_Acc], 'Sens': [Bestloss_Test_Sens], 'Spec': [Bestloss_Test_Spec], 'JW5': [Bestloss_Test_jw5], 'JW6': [Bestloss_Test_jw6]}), ignore_index=True)
    result = result.append(pd.DataFrame(
        {'Stage': 'Bestloss_Test_unique', 'AUC': [Bestloss_Test_Auc_bestepoch], 'ACC': [Bestloss_Test_Acc_bestepoch], 'Sens': [Bestloss_Test_Sens_bestepoch], 'Spec': [Bestloss_Test_Spec_bestepoch], 'JW5': [Bestloss_Test_jw5_bestepoch], 'JW6': [Bestloss_Test_jw6_bestepoch]}), ignore_index=True)


    result = result.append(pd.DataFrame(
        {'Stage': 'Bestauc_Train', 'AUC': [Bestauc_Train_Auc], 'ACC': [Bestauc_Train_Acc], 'Sens': [Bestauc_Train_Sens], 'Spec': [Bestauc_Train_Spec], 'JW5': [Bestauc_Train_jw5], 'JW6': [Bestauc_Train_jw6]}), ignore_index=True)
    result = result.append(pd.DataFrame(
        {'Stage': 'Bestauc_Val', 'AUC': [Bestauc_Val_Auc], 'ACC': [Bestauc_Val_Acc], 'Sens': [Bestauc_Val_Sens], 'Spec': [Bestauc_Val_Spec], 'JW5': [Bestauc_Val_jw5], 'JW6': [Bestauc_Val_jw6]}), ignore_index=True)
    result = result.append(pd.DataFrame(
        {'Stage': 'Bestauc_Test', 'AUC': [Bestauc_Test_Auc], 'ACC': [Bestauc_Test_Acc], 'Sens': [Bestauc_Test_Sens], 'Spec': [Bestauc_Test_Spec], 'JW5': [Bestauc_Test_jw5], 'JW6': [Bestauc_Test_jw6]}), ignore_index=True)
    result = result.append(pd.DataFrame(
        {'Stage': 'Bestauc_Test_unique', 'AUC': [Bestauc_Test_Auc_bestepoch], 'ACC': [Bestauc_Test_Acc_bestepoch], 'Sens': [Bestauc_Test_Sens_bestepoch], 'Spec': [Bestauc_Test_Spec_bestepoch], 'JW5': [Bestauc_Test_jw5_bestepoch], 'JW6': [Bestauc_Test_jw6_bestepoch]}), ignore_index=True)

    result = result.append(pd.DataFrame(
        {'Stage': 'Bestjw5_Train', 'AUC': [Bestjw5_Train_Auc], 'ACC': [Bestjw5_Train_Acc], 'Sens': [Bestjw5_Train_Sens], 'Spec': [Bestjw5_Train_Spec], 'JW5': [Bestjw5_Train_jw5], 'JW6': [Bestjw5_Train_jw6]}), ignore_index=True)
    result = result.append(pd.DataFrame(
        {'Stage': 'Bestjw5_Val', 'AUC': [Bestjw5_Val_Auc], 'ACC': [Bestjw5_Val_Acc], 'Sens': [Bestjw5_Val_Sens], 'Spec': [Bestjw5_Val_Spec], 'JW5': [Bestjw5_Val_jw5], 'JW6': [Bestjw5_Val_jw6]}), ignore_index=True)
    result = result.append(pd.DataFrame(
        {'Stage': 'Bestjw5_Test', 'AUC': [Bestjw5_Test_Auc], 'ACC': [Bestjw5_Test_Acc], 'Sens': [Bestjw5_Test_Sens], 'Spec': [Bestjw5_Test_Spec], 'JW5': [Bestjw5_Test_jw5], 'JW6': [Bestjw5_Test_jw6]}), ignore_index=True)
    result = result.append(pd.DataFrame(
        {'Stage': 'Bestjw5_Test_unique', 'AUC': [Bestjw5_Test_Auc_bestepoch], 'ACC': [Bestjw5_Test_Acc_bestepoch], 'Sens': [Bestjw5_Test_Sens_bestepoch], 'Spec': [Bestjw5_Test_Spec_bestepoch],'JW5': [Bestjw5_Test_jw5_bestepoch], 'JW6': [Bestjw5_Test_jw6_bestepoch]}), ignore_index=True)

    result = result.append(pd.DataFrame(
        {'Stage': 'Bestjw6_Train', 'AUC': [Bestjw6_Train_Auc], 'ACC': [Bestjw6_Train_Acc], 'Sens': [Bestjw6_Train_Sens], 'Spec': [Bestjw6_Train_Spec], 'JW5': [Bestjw6_Train_jw5], 'JW6': [Bestjw6_Train_jw6]}), ignore_index=True)
    result = result.append(pd.DataFrame(
        {'Stage': 'Bestjw6_Val', 'AUC': [Bestjw6_Val_Auc], 'ACC': [Bestjw6_Val_Acc], 'Sens': [Bestjw6_Val_Sens], 'Spec': [Bestjw6_Val_Spec], 'JW5': [Bestjw6_Val_jw5], 'JW6': [Bestjw6_Val_jw6]}), ignore_index=True)
    result = result.append(pd.DataFrame(
        {'Stage': 'Bestjw6_Test', 'AUC': [Bestjw6_Test_Auc], 'ACC': [Bestjw6_Test_Acc], 'Sens': [Bestjw6_Test_Sens], 'Spec': [Bestjw6_Test_Spec], 'JW5': [Bestjw6_Test_jw5], 'JW6': [Bestjw6_Test_jw6]}), ignore_index=True)
    result = result.append(pd.DataFrame(
        {'Stage': 'Bestjw6_Test_unique', 'AUC': [Bestjw6_Test_Auc_bestepoch], 'ACC': [Bestjw6_Test_Acc_bestepoch], 'Sens': [Bestjw6_Test_Sens_bestepoch], 'Spec': [Bestjw6_Test_Spec_bestepoch], 'JW5': [Bestjw6_Test_jw5_bestepoch], 'JW6': [Bestjw6_Test_jw6_bestepoch]}), ignore_index=True)
    result.to_csv(save_result)


def save_temp(result_bestloss_train_folds, result_bestloss_val_folds, result_bestloss_test_folds,
              result_bestauc_train_folds, result_bestauc_val_folds, result_bestauc_test_folds,
              result_bestjw5_train_folds, result_bestjw5_val_folds, result_bestjw5_test_folds,
              result_bestjw6_train_folds, result_bestjw6_val_folds, result_bestjw6_test_folds,
              savefile):

    with pd.ExcelWriter(savefile, engine='xlsxwriter') as writer:

        for fold in range(len(result_bestloss_train_folds)):
            # [loss_train, auc_train, acc_train, sens_train, spec_train]
            result_temp = pd.DataFrame(columns=('Stage', 'AUC', 'ACC', 'Sens', 'Spec', 'jw0.5', 'jw0.6'))

            # bestloss
            result_temp = result_temp.append(pd.DataFrame(
                {'Stage': 'Bestloss_Train', 'AUC': [result_bestloss_train_folds[fold][1]],
                 'ACC': [result_bestloss_train_folds[fold][2]],
                 'Sens': [result_bestloss_train_folds[fold][3]], 'Spec': [result_bestloss_train_folds[fold][4]],
                 'jw0.5': [0.5*result_bestloss_train_folds[fold][3]+0.5*result_bestloss_train_folds[fold][4]],
                 'jw0.6': [0.6*result_bestloss_train_folds[fold][3]+0.4*result_bestloss_train_folds[fold][4]]}), ignore_index=True)
            result_temp = result_temp.append(pd.DataFrame(
                {'Stage': 'Bestloss_Val', 'AUC': [result_bestloss_val_folds[fold][1]],
                 'ACC': [result_bestloss_val_folds[fold][2]],
                 'Sens': [result_bestloss_val_folds[fold][3]], 'Spec': [result_bestloss_val_folds[fold][4]],
                 'jw0.5': [0.5 * result_bestloss_val_folds[fold][3] + 0.5 * result_bestloss_val_folds[fold][4]],
                 'jw0.6': [0.6 * result_bestloss_val_folds[fold][3] + 0.4 * result_bestloss_val_folds[fold][4]]}),ignore_index=True)
            result_temp = result_temp.append(pd.DataFrame(
                {'Stage': 'Bestloss_Test', 'AUC': [result_bestloss_test_folds[fold][1]],
                 'ACC': [result_bestloss_test_folds[fold][2]],
                 'Sens': [result_bestloss_test_folds[fold][3]], 'Spec': [result_bestloss_test_folds[fold][4]],
                 'jw0.5': [0.5 * result_bestloss_test_folds[fold][3] + 0.5 * result_bestloss_test_folds[fold][4]],
                 'jw0.6': [0.6 * result_bestloss_test_folds[fold][3] + 0.4 * result_bestloss_test_folds[fold][4]]}),ignore_index=True)
            # bestauc
            result_temp = result_temp.append(pd.DataFrame(
                {'Stage': 'Bestauc_Train', 'AUC': [result_bestauc_train_folds[fold][1]],
                 'ACC': [result_bestauc_train_folds[fold][2]],
                 'Sens': [result_bestauc_train_folds[fold][3]], 'Spec': [result_bestauc_train_folds[fold][4]],
                 'jw0.5': [0.5 * result_bestauc_train_folds[fold][3] + 0.5 * result_bestauc_train_folds[fold][4]],
                 'jw0.6': [0.6 * result_bestauc_train_folds[fold][3] + 0.4 * result_bestauc_train_folds[fold][4]]}),
                ignore_index=True)
            result_temp = result_temp.append(pd.DataFrame(
                {'Stage': 'Bestauc_Val', 'AUC': [result_bestauc_val_folds[fold][1]],
                 'ACC': [result_bestauc_val_folds[fold][2]],
                 'Sens': [result_bestauc_val_folds[fold][3]], 'Spec': [result_bestauc_val_folds[fold][4]],
                 'jw0.5': [0.5 * result_bestauc_val_folds[fold][3] + 0.5 * result_bestauc_val_folds[fold][4]],
                 'jw0.6': [0.6 * result_bestauc_val_folds[fold][3] + 0.4 * result_bestauc_val_folds[fold][4]]}),
                ignore_index=True)
            result_temp = result_temp.append(pd.DataFrame(
                {'Stage': 'Bestauc_Test', 'AUC': [result_bestauc_test_folds[fold][1]],
                 'ACC': [result_bestauc_test_folds[fold][2]],
                 'Sens': [result_bestauc_test_folds[fold][3]], 'Spec': [result_bestauc_test_folds[fold][4]],
                 'jw0.5': [0.5 * result_bestauc_test_folds[fold][3] + 0.5 * result_bestauc_test_folds[fold][4]],
                 'jw0.6': [0.6 * result_bestauc_test_folds[fold][3] + 0.4 * result_bestauc_test_folds[fold][4]]}),
                ignore_index=True)
            # best_jw0.5
            result_temp = result_temp.append(pd.DataFrame(
                {'Stage': 'Bestjw5_Train', 'AUC': [result_bestjw5_train_folds[fold][1]],
                 'ACC': [result_bestjw5_train_folds[fold][2]],
                 'Sens': [result_bestjw5_train_folds[fold][3]], 'Spec': [result_bestjw5_train_folds[fold][4]],
                 'jw0.5': [0.5 * result_bestjw5_train_folds[fold][3] + 0.5 * result_bestjw5_train_folds[fold][4]],
                 'jw0.6': [0.6 * result_bestjw5_train_folds[fold][3] + 0.4 * result_bestjw5_train_folds[fold][4]]}),
                ignore_index=True)
            result_temp = result_temp.append(pd.DataFrame(
                {'Stage': 'Bestjw5_Val', 'AUC': [result_bestjw5_val_folds[fold][1]],
                 'ACC': [result_bestjw5_val_folds[fold][2]],
                 'Sens': [result_bestjw5_val_folds[fold][3]], 'Spec': [result_bestjw5_val_folds[fold][4]],
                 'jw0.5': [0.5 * result_bestjw5_val_folds[fold][3] + 0.5 * result_bestjw5_val_folds[fold][4]],
                 'jw0.6': [0.6 * result_bestjw5_val_folds[fold][3] + 0.4 * result_bestjw5_val_folds[fold][4]]}),
                ignore_index=True)
            result_temp = result_temp.append(pd.DataFrame(
                {'Stage': 'Bestjw5_Test', 'AUC': [result_bestjw5_test_folds[fold][1]],
                 'ACC': [result_bestjw5_test_folds[fold][2]],
                 'Sens': [result_bestjw5_test_folds[fold][3]], 'Spec': [result_bestjw5_test_folds[fold][4]],
                 'jw0.5': [0.5 * result_bestjw5_test_folds[fold][3] + 0.5 * result_bestjw5_test_folds[fold][4]],
                 'jw0.6': [0.6 * result_bestjw5_test_folds[fold][3] + 0.4 * result_bestjw5_test_folds[fold][4]]}),
                ignore_index=True)
            # best_jw0.6
            result_temp = result_temp.append(pd.DataFrame(
                {'Stage': 'Bestjw6_Train', 'AUC': [result_bestjw6_train_folds[fold][1]],
                 'ACC': [result_bestjw6_train_folds[fold][2]],
                 'Sens': [result_bestjw6_train_folds[fold][3]], 'Spec': [result_bestjw6_train_folds[fold][4]],
                 'jw0.5': [0.5 * result_bestjw6_train_folds[fold][3] + 0.5 * result_bestjw6_train_folds[fold][4]],
                 'jw0.6': [0.6 * result_bestjw6_train_folds[fold][3] + 0.4 * result_bestjw6_train_folds[fold][4]]}),
                ignore_index=True)
            result_temp = result_temp.append(pd.DataFrame(
                {'Stage': ['Bestjw6_Val'], 'AUC': [result_bestjw6_val_folds[fold][1]],
                 'ACC': [result_bestjw6_val_folds[fold][2]],
                 'Sens': [result_bestjw6_val_folds[fold][3]], 'Spec': [result_bestjw6_val_folds[fold][4]],
                 'jw0.5': [0.5 * result_bestjw6_val_folds[fold][3] + 0.5 * result_bestjw6_val_folds[fold][4]],
                 'jw0.6': [0.6 * result_bestjw6_val_folds[fold][3] + 0.4 * result_bestjw6_val_folds[fold][4]]}),
                ignore_index=True)
            result_temp = result_temp.append(pd.DataFrame(
                {'Stage': 'Bestjw6_Test', 'AUC': [result_bestjw6_test_folds[fold][1]],
                 'ACC': [result_bestjw6_test_folds[fold][2]],
                 'Sens': [result_bestjw6_test_folds[fold][3]], 'Spec': [result_bestjw6_test_folds[fold][4]],
                 'jw0.5': [0.5 * result_bestjw6_test_folds[fold][3] + 0.5 * result_bestjw6_test_folds[fold][4]],
                 'jw0.6': [0.6 * result_bestjw6_test_folds[fold][3] + 0.4 * result_bestjw6_test_folds[fold][4]]}),
                ignore_index=True)

            result_temp.to_excel(writer, 'Fold'+str(fold))