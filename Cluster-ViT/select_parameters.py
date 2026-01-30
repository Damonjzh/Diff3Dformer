import datetime
import argparse
import os

def write_parameter(paras, savepath):

    open_type = 'a' if os.path.exists(savepath) else 'w'
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    with open(savepath, open_type) as f:
        f.write(now + '\n\n')
        for arg in vars(paras):
            f.write('{}: {}\n'.format(arg, getattr(paras, arg)))
        f.write('\n')
        f.close()



def pld_mortality_vit_parameter():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    parser.add_argument('--position_embedding', default='3Dlearned', type=str,
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true', default=False)
    parser.add_argument('--pretrained_path', default='', type=str, help="path of pretrained model")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')

    parser.add_argument('--output_dir', default='./',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    # parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--kfoldNum', default=5, type=int)
    parser.add_argument('--dataDir', type=str, help='path of the data')
    parser.add_argument('--externalDataDir', type=str, help='path of the external test data')

    # distributed training parameters
    parser.add_argument('--world_size', default=4, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', action='store_true', default=False)

    # cluster parameters
    parser.add_argument('--group_Q', action='store_true', default=False)
    parser.add_argument('--group_K', action='store_true', default=False)
    parser.add_argument('--cuda-devices', default=None)
    parser.add_argument('--max_num_cluster', default=64, type=int)
    parser.add_argument('--sequence_len', default=15000, type=int)

    # gridsearch parameters
    parser.add_argument('--withPosEmbedding', action='store_true', default=False)
    parser.add_argument('--seq_pool', action='store_true', default=False,
                        help='use attention pooling layer for aggregating patch risk score')
    parser.add_argument('--withLN', action='store_true', default=False)
    parser.add_argument('--withEmbeddingPreNorm', action='store_true', default=False,
                        help='Pre-normalize the patch representation before feeding them into ViT')
    parser.add_argument('--input_pool', action='store_true', default=False)
    parser.add_argument('--mixUp', action='store_true', default=False)
    parser.add_argument('--SAM', action='store_true', default=False)
    parser.add_argument('--augmented', action='store_true', default=False)

    #original
    parser.add_argument('--perform_statistics', action='store_true')
    parser.add_argument('--use_clinical', action='store_true')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--continue_train', action='store_true')

    parser.add_argument('--patient_died_ct_csv', dest="patient_died_ct_csv",
                        default='../dataset/data_cam/patients_enrol_list/finalstatus_patient_died_ct_unique.csv')
    parser.add_argument('--patient_survived_ct_csv', dest="patient_survived_ct_csv",
                        default='../dataset/data_cam/patients_enrol_list/finalstatus_patient_survived_ct_unique.csv')
    parser.add_argument('--patiens_info_csv_train', dest="patiens_info_csv_train",
                        default='../dataset/data_osic/patients_enrol_list/OSIC_info.csv')
    parser.add_argument('--patiens_info_csv_test', dest="patiens_info_csv_test",
                        default='../dataset/data_australia/patients_enrol_list/Australia_info.csv')
    parser.add_argument('--expname', dest="expname",
                        default='test_mortality')


    # parameter for dataset
    parser.add_argument('--dataloader', dest="dataloader",
                        default='2D_montage_OSIC_AUS')
    # not used actually
    parser.add_argument('--datapath_coronal', dest="datapath_coronal",
                        default='../dataset/OSIC/CT_img_2Dmontage/')
    parser.add_argument('--datapath_train', dest="datapath_train",  ###############?????????????
                        default='../dataset/OSIC/CT_img_2Dmontage/')
    parser.add_argument('--datapath_mask_train', dest="datapath_mask_train",  ###############?????????????
                        default='../dataset/OSIC/CT_img_2Dmontagemask/')
    parser.add_argument('--datapath_test', dest="datapath_test",  ###############?????????????
                        default='../dataset/AUSTRALIA/CT_img_2Dmontage/')
    parser.add_argument('--datapath_mask_test', dest="datapath_mask_test",  ###############?????????????
                        default='../dataset/AUSTRALIA/CT_img_2Dmontagemask/')
    parser.add_argument('--data_clinical', dest="data_clinical",
                        default='../Documents/xai/reproduce_weakly/cleanup/dataset/data_cam/patients_enrol_list/patient_all_info.csv')

    parser.add_argument('--aug_train', dest="aug_train", nargs="+",
                        default=[1, 30], type=int)
    parser.add_argument('--aug_val', dest="aug_val", nargs="+",
                        default=[1, 1], type=int)
    parser.add_argument('--aug_test', dest="aug_test", nargs="+",
                        default=[1, 1], type=int)
    parser.add_argument('--clinical_category', help='delimited list input', type=str, default="")
    parser.add_argument('--clinical_continuous', help='delimited list input', type=str, default="")
    # training
    parser.add_argument('--model_name', dest="model_name",
                        default='densenet')
    parser.add_argument('--optimizer', dest="optimizer", default="sgd")
    parser.add_argument('--bs', dest="bs", default=8, type=int)  # default=16
    parser.add_argument('--drop_rate', dest="drop_rate", default=0, type=float)
    parser.add_argument('--epoch', dest="epoch", default=50, type=int)
    # parser.add_argument('--device', dest="device", default="cuda", type=str)
    parser.add_argument('--gpu_num', dest="gpu_num", default="0", type=str)

    # loss
    parser.add_argument('--loss_name', dest="loss_name",
                        default='Focal')  # Focal, MSE, CrossEntropyLoss
    parser.add_argument('--alpha', dest="alpha",
                        default=1, type=float)
    parser.add_argument('--gamma', dest="gamma",
                        default=1, type=float)
    parser.add_argument('--in_channel', dest="in_channel", default=3, type=int)
    parser.add_argument('--len_clinical', dest="len_clinical", default=0, type=int)

    # ooptimizer
    parser.add_argument('--opt_name', dest="opt_name",
                        default='Adam')
    # parser.add_argument('--lr', dest="lr",
    #                     default=1e-6, type=float)
    parser.add_argument('--jw_ratio', dest="jw_ratio", default=0.5, type=float)

    parser.add_argument('--use_only_clinical', action='store_true')
    parser.add_argument('--use_only_ct_axial', action='store_true')
    parser.add_argument('--use_only_ct_coronal', action='store_true')
    parser.add_argument('--use_fuse_clinical_axial', action='store_true')
    parser.add_argument('--use_fuse_axial_coronal', action='store_true')
    parser.add_argument('--use_BI', action='store_true')
    parser.add_argument('--use_fix', action='store_true')
    parser.add_argument('--use_MI', action='store_true')
    parser.add_argument('--use_CONF', action='store_true')
    parser.add_argument('--use_ATTEN', action='store_true')
    parser.add_argument('--use_SPARSE', action='store_true')
    parser.add_argument('--use_selfpretrain', action='store_true')
    parser.add_argument('--use_similarity_inter', action='store_true')

    parser.add_argument('--test_xai', action='store_true')
    parser.add_argument('--fuseadd', action='store_true')
    parser.add_argument('--fuseend', action='store_true')

    parser.add_argument('--dim_bottleneck', dest="dim_bottleneck", default=256, type=int)

    parser.add_argument('--weight_ob', dest="weight_ob",
                        default=1, type=float)
    parser.add_argument('--weight_re', dest="weight_re",
                        default=1, type=float)
    parser.add_argument('--weight_axial', dest="weight_axial",
                        default=1, type=float)
    parser.add_argument('--weight_cli_cro', dest="weight_cli_cro",
                        default=1, type=float)
    parser.add_argument('--weight_MI', dest="weight_MI",
                        default=1, type=float)
    parser.add_argument('--weight_IB', dest="weight_IB",
                        default=1, type=float)
    parser.add_argument('--weight_CONF', dest="weight_CONF",
                        default=1, type=float)
    parser.add_argument('--weight_SPARSE', dest="weight_SPARSE",
                        default=0.5, type=float)

    args = parser.parse_args()
    return args