# import timm
import torch
import torch.nn as nn
import sys
sys.path.append("../..")

def select_loss(opt):

    if opt.loss_name == 'MSE':
        loss = nn.MSELoss()
    elif opt.loss_name == 'CrossEntropyLoss':
        loss = nn.CrossEntropyLoss()
    elif opt.loss_name == 'Focal':
        from loss.loss_focal import FocalLoss
        loss = FocalLoss(class_num=2, alpha=opt.alpha, gamma=opt.gamma, size_average=True)
    elif opt.loss_name == 'BI':
        from loss.loss_BI import Loss_BI
        # from loss.loss_focal import FocalLoss
        loss = Loss_BI(opt)
        # loss2 = FocalLoss(class_num=2, alpha=opt.alpha, gamma=opt.gamma, size_average=True)
        # loss = opt.VSD_loss * loss1 + loss2
    elif opt.loss_name == 'BI+mutual':
        from loss.loss_BI import Loss_BI
        # from loss.loss_focal import FocalLoss
        # loss = Loss_BI(opt)[0] +
        # loss2 = FocalLoss(class_num=2, alpha=opt.alpha, gamma=opt.gamma, size_average=True)
        # loss = opt.VSD_loss * loss1 + loss2
    return loss
