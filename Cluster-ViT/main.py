import os

import pandas as pd

gpu_num = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_num
import torch
print(torch.cuda.device_count())
import argparse
import datetime
import json

import random
from re import A
import time
from pathlib import Path


from torch.utils.data.dataset import Subset
from datasets.MyData import MyDataset
import numpy as np

from torch.utils.data import DataLoader, DistributedSampler, random_split
from sklearn.model_selection import KFold
import util.misc as utils

from models.engine_DMIB import evaluate_DMIB, train_one_epoch_SAM_DMIB, test, save_final, save_temp
from models import build_model_new
from models.sam import SAM
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split, StratifiedKFold

from select_parameters import *
from torch.utils.data import DataLoader
from util.trainer import *
import select_loss
import select_optimizer
import select_parameters

setup_seed(20)

args = pld_mortality_vit_parameter()

if not args.clinical_category == '':
    args.clinical_category = [str(item) for item in args.clinical_category.split(',')]
else:
    args.clinical_category = []

if not args.clinical_continuous == '':
    args.clinical_continuous = [str(item) for item in args.clinical_continuous.split(',')]
else:
    args.clinical_continuous = []

args.len_clinical = len(args.clinical_category) + len(args.clinical_continuous)
print("clinical data: {}".format(args.len_clinical))
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_num

save_location = './experiments/'
save_logdir = save_location + args.expname + '/logdir/'  # fold1/2/3/4/5

save_model = save_location + args.expname + '/model/'  # fold1/2/3/4/5, best_auc, best_loss
save_param = save_location + args.expname + '/para.txt'
save_result = './result_exp/' + args.expname + '/result.csv'

save_result_ob = save_location + args.expname + '/result_ob.csv'
save_result_temp_ob = save_location + args.expname + '/result_temp_ob.xlsx'
save_result_re = save_location + args.expname + '/result_re.csv'
save_result_temp_re = save_location + args.expname + '/result_temp_re.xlsx'

os.makedirs(save_logdir, exist_ok=True)
os.makedirs(save_model, exist_ok=True)
write_parameter(args, save_param)

'''
Split the train dataset by 10-fold: select the model with largest AUC on the validation for the test set .
'''

result_bestloss_train_folds = []
result_bestloss_val_folds = []
result_bestloss_test_folds = []
result_bestauc_train_folds = []
result_bestauc_val_folds = []
result_bestauc_test_folds = []
result_bestjw5_train_folds = []
result_bestjw5_val_folds = []
result_bestjw5_test_folds = []
result_bestjw6_train_folds = []
result_bestjw6_val_folds = []
result_bestjw6_test_folds = []

result_bestloss_train_folds_ob = []
result_bestloss_val_folds_ob = []
result_bestloss_test_folds_ob = []
result_bestauc_train_folds_ob = []
result_bestauc_val_folds_ob = []
result_bestauc_test_folds_ob = []
result_bestjw5_train_folds_ob = []
result_bestjw5_val_folds_ob = []
result_bestjw5_test_folds_ob = []
result_bestjw6_train_folds_ob = []
result_bestjw6_val_folds_ob = []
result_bestjw6_test_folds_ob = []

result_bestloss_train_folds_re = []
result_bestloss_val_folds_re = []
result_bestloss_test_folds_re = []
result_bestauc_train_folds_re = []
result_bestauc_val_folds_re = []
result_bestauc_test_folds_re = []
result_bestjw5_train_folds_re = []
result_bestjw5_val_folds_re = []
result_bestjw5_test_folds_re = []
result_bestjw6_train_folds_re = []
result_bestjw6_val_folds_re = []
result_bestjw6_test_folds_re = []

num_fold = 0
start_num_fold = 0
# args.continue_train = True
if args.continue_train:
    start_num_fold, \
        result_bestloss_train_folds_ob, result_bestloss_val_folds_ob, result_bestloss_test_folds_ob, \
        result_bestauc_train_folds_ob, result_bestauc_val_folds_ob, result_bestauc_test_folds_ob, \
        result_bestjw5_train_folds_ob, result_bestjw5_val_folds_ob, result_bestjw5_test_folds_ob, \
        result_bestjw6_train_folds_ob, result_bestjw6_val_folds_ob, result_bestjw6_test_folds_ob \
        = load_temp(save_result_temp_ob)

    start_num_fold, \
        result_bestloss_train_folds_re, result_bestloss_val_folds_re, result_bestloss_test_folds_re, \
        result_bestauc_train_folds_re, result_bestauc_val_folds_re, result_bestauc_test_folds_re, \
        result_bestjw5_train_folds_re, result_bestjw5_val_folds_re, result_bestjw5_test_folds_re, \
        result_bestjw6_train_folds_re, result_bestjw6_val_folds_re, result_bestjw6_test_folds_re \
        = load_temp(save_result_temp_re)

num_fold = 0
start_num_fold = 0

train_csv = pd.read_excel('./DnR/inference/ccccii_slices/train_ncp_cp.xlsx')
train_index = train_csv['patientid'].to_numpy()
val_csv = pd.read_excel('./DnR/inference/ccccii_slices/Test_ncp_cp.xlsx')
val_index = val_csv['patientid'].to_numpy()
test_csv = pd.read_excel('./DnR/inference/ccccii_slices/val_ncp_cp.xlsx')
test_index = test_csv['patientid'].to_numpy()

dataset_train = MyDataset(root_dir=args.dataDir, sequence_len=args.sequence_len, max_num_cluster=args.max_num_cluster,
                       status='train', input_pool=args.input_pool, data_index=train_index)
dataset_val = MyDataset(root_dir=args.dataDir, sequence_len=args.sequence_len, max_num_cluster=args.max_num_cluster,
                     status='val', input_pool=args.input_pool, data_index=val_index)
dataset_test = MyDataset(root_dir=args.dataDir, sequence_len=args.sequence_len, max_num_cluster=args.max_num_cluster,
                      status='test', input_pool=args.input_pool, data_index=test_index)
dataset_train_all = MyDataset(root_dir=args.dataDir, sequence_len=args.sequence_len, max_num_cluster=args.max_num_cluster,
                           status='test', input_pool=args.input_pool)

# utils.init_distributed_mode(args)
print(args)
device = torch.device("cuda:0")
print('gpu_num: ', torch.cuda.device_count())

# fix the seed for reproducibility
seed = args.seed + utils.get_rank()
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

model, criterion = build_model_new(args)
model.to(device)
model.cuda()

model_without_ddp = model
if args.distributed:
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    model = nn.DataParallel(model).to(device)
    model_without_ddp = model.module

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    model_without_ddp = model.module

n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('number of params:', n_parameters)

if args.SAM:
    base_optimizer = torch.optim.Adam  # define an optimizer for the "sharpness-aware" update
    optimizer_SAM = SAM(model_without_ddp.parameters(), base_optimizer, lr=args.lr, weight_decay=args.weight_decay)
    optimizer = optimizer_SAM
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
else:
    optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

best_loss = 1e12
tb_writer = SummaryWriter(save_location + args.expname)

if args.distributed:
    sampler_train = DistributedSampler(dataset_train)
    sampler_val = DistributedSampler(dataset_val, shuffle=False)
    sampler_test = DistributedSampler(dataset_test, shuffle=False)
else:
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)

batch_sampler_train = torch.utils.data.BatchSampler(
    sampler_train, args.batch_size, drop_last=False)

data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                               collate_fn=None, num_workers=args.num_workers)
data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                             drop_last=False, num_workers=args.num_workers)
data_loader_test = DataLoader(dataset_test, args.batch_size, sampler=sampler_test,
                              drop_last=False, num_workers=args.num_workers)
lossfun = select_loss(args)
# model = torch.nn.DataParallel(model)

output_dir = Path(args.output_dir)
if args.kfoldNum > 1:
    output_dir = output_dir / f'fold{num_fold}'
    output_dir.mkdir(parents=True, exist_ok=True)

if args.resume:
    checkpoint = torch.load('./experiments/ccccii/model/best_jw5_ob.pt')
    model_without_ddp.load_state_dict(checkpoint['model'])
    if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
if args.pretrained_path != '':
    pretrainedModel = torch.load(args.pretrained_path)
    model_without_ddp.load_state_dict(pretrainedModel['model_state_dict'], strict=False)
if args.eval:
    # val_stats = test(model_without_ddp, criterion, data_loader_test,data_loader_train, device, args.output_dir, status='test')
    val_stats = evaluate_DMIB(model_without_ddp, criterion, data_loader_test, device, args.output_dir,
                     status='test')
print(f"fold {num_fold}/{args.kfoldNum} Start training")
start_time = time.time()

save_model_fold = save_model
os.makedirs(save_model_fold, exist_ok=True)

# logging setup
# wandb.watch(model)
train_scheduler = None
test_scheduler = None

loss_val_best_ob = 9999
auc_val_best_ob = -9999
jw5_val_best_ob = -9999
jw6_val_best_ob = -9999
loss_val_best_re = 9999
auc_val_best_re = -9999
jw5_val_best_re = -9999
jw6_val_best_re = -9999
loss_val_best = 9999
auc_val_best = -9999
jw_val_best5 = -9999
jw_val_best6 = -9999

for epoch in range(0, args.epochs):

    write_epoch = epoch
    model = model_without_ddp
    # Train
    train_loss, train_auc, train_acc, train_spec, train_sens, train_jw5, train_jw6 = train_one_epoch_SAM_DMIB(
        model, criterion, data_loader_train, optimizer, device, epoch, num_fold, tb_writer,
        args.clip_max_norm, mixUp=args.mixUp)

    print('train_loss', train_loss)
    tb_writer.add_scalar('train/train_loss',train_loss, write_epoch)
    tb_writer.add_scalar('train/train_auc',train_auc, write_epoch)
    tb_writer.add_scalar('train/train_acc',train_acc, write_epoch)
    tb_writer.add_scalar('train/train_spec',train_spec, write_epoch)
    tb_writer.add_scalar('train/train_sens',train_sens, write_epoch)
    tb_writer.add_scalar('train/train_jw5',train_jw5, write_epoch)
    tb_writer.add_scalar('train/train_jw6',train_jw6, write_epoch)


    lr_scheduler.step()

    val_loss, val_auc, val_acc, val_spec, val_sens, val_jw5, val_jw6 = evaluate_DMIB(
        model, criterion, data_loader_val, device, args.output_dir, 'Validation')
    tb_writer.add_scalar('val/val_loss', val_loss, write_epoch)
    tb_writer.add_scalar('val/val_auc', val_auc, write_epoch)
    tb_writer.add_scalar('val/val_acc', val_acc, write_epoch)
    tb_writer.add_scalar('val/val_spec', val_spec, write_epoch)
    tb_writer.add_scalar('val/val_sens', val_sens, write_epoch)
    tb_writer.add_scalar('val/val_jw5', val_jw5, write_epoch)
    tb_writer.add_scalar('val/val_jw6', val_jw6, write_epoch)

    print('val_loss', val_loss)

    test_loss, test_auc, test_acc, test_spec, test_sens, test_jw5, test_jw6 = evaluate_DMIB(
        model, criterion, data_loader_test, device, args.output_dir, 'Test',num_fold)
    tb_writer.add_scalar('test/test_loss', test_loss, write_epoch)
    tb_writer.add_scalar('test/test_auc', test_auc, write_epoch)
    tb_writer.add_scalar('test/test_acc', test_acc, write_epoch)
    tb_writer.add_scalar('test/test_spec', test_spec, write_epoch)
    tb_writer.add_scalar('test/test_sens', test_sens, write_epoch)
    tb_writer.add_scalar('test/test_jw5', test_jw5, write_epoch)
    tb_writer.add_scalar('test/test_jw6', test_jw6, write_epoch)

    print('test_loss', test_loss)


    if val_loss < loss_val_best_ob:
        loss_val_best_ob = val_loss
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict()},
                   save_model_fold + 'best_loss_ob.pt')
        result_bestloss_train = [train_loss, train_auc, train_acc, train_sens, train_spec,
                                    train_jw5, train_jw6]
        result_bestloss_val = [val_loss, val_auc, val_acc, val_sens, val_spec, val_jw5,
                                  val_jw6]
        result_bestloss_test = [test_loss, test_auc, test_acc, test_sens, test_spec, test_jw5,
                                test_jw6]
    if val_auc > auc_val_best_ob:
        auc_val_best_ob = val_auc
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict()},
                   save_model_fold + 'best_auc_ob.pt')
        result_bestauc_train = [train_auc, train_auc, train_acc, train_sens, train_spec,
                                   train_jw5, train_jw6]
        result_bestauc_val = [val_loss, val_auc, val_acc, val_sens, val_spec, val_jw5,
                                 val_jw6]
        result_bestauc_test = [test_loss, test_auc, test_acc, test_sens, test_spec, test_jw5,
                                test_jw6]
    if val_jw5 > jw5_val_best_ob:
        jw5_val_best_ob = val_jw5
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict()},
                   save_model_fold + 'best_jw5_ob.pt')
        result_bestjw5_train = [train_auc, train_auc, train_acc, train_sens, train_spec,
                                   train_jw5, train_jw6]
        result_bestjw5_val = [val_loss, val_auc, val_acc, val_sens, val_spec, val_jw5,
                                 val_jw6]
        result_bestjw5_test = [test_loss, test_auc, test_acc, test_sens, test_spec, test_jw5,
                                test_jw6]
    if val_jw6 > jw6_val_best_ob:
        jw6_val_best_ob = val_jw6
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict()},
                   save_model_fold + 'best_jw6_ob.pt')
        result_bestjw6_train = [train_auc, train_auc, train_acc, train_sens, train_spec,
                                   train_jw5, train_jw6]
        result_bestjw6_val = [val_loss, val_auc, val_acc, val_sens, val_spec, val_jw5,
                                 val_jw6]
        result_bestjw6_test = [test_loss, test_auc, test_acc, test_sens, test_spec, test_jw5,
                                test_jw6]


    if epoch == args.epochs - 1:

        result_bestloss_train_folds_ob.append(result_bestloss_train)
        result_bestloss_val_folds_ob.append(result_bestloss_val)
        result_bestloss_test_folds_ob.append(result_bestloss_test)
        result_bestauc_train_folds_ob.append(result_bestauc_train)
        result_bestauc_val_folds_ob.append(result_bestauc_val)
        result_bestauc_test_folds_ob.append(result_bestauc_test)
        result_bestjw5_train_folds_ob.append(result_bestjw5_train)
        result_bestjw5_val_folds_ob.append(result_bestjw5_val)
        result_bestjw5_test_folds_ob.append(result_bestjw5_test)
        result_bestjw6_train_folds_ob.append(result_bestjw6_train)
        result_bestjw6_val_folds_ob.append(result_bestjw6_val)
        result_bestjw6_test_folds_ob.append(result_bestjw6_test)

        save_temp_vit(result_bestloss_train_folds_ob, result_bestloss_val_folds_ob, result_bestloss_test_folds_ob, \
                  result_bestauc_train_folds_ob, result_bestauc_val_folds_ob, result_bestauc_test_folds_ob, \
                  result_bestjw5_train_folds_ob, result_bestjw5_val_folds_ob, result_bestjw5_test_folds_ob, \
                  result_bestjw6_train_folds_ob, result_bestjw6_val_folds_ob, result_bestjw6_test_folds_ob,
                  save_result_temp_ob)

        num_fold = num_fold + 1

save_final_vit(result_bestloss_train_folds_ob, result_bestloss_val_folds_ob, result_bestloss_test_folds_ob,
               result_bestauc_train_folds_ob, result_bestauc_val_folds_ob, result_bestauc_test_folds_ob,
               result_bestjw5_train_folds_ob, result_bestjw5_val_folds_ob, result_bestjw5_test_folds_ob,
               result_bestjw6_train_folds_ob, result_bestjw6_val_folds_ob, result_bestjw6_test_folds_ob, save_result_ob)

