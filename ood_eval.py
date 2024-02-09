from utils import log
import torch
import torch.nn as nn
import time
import csv
import numpy as np

from utils.test_utils import get_measures
import os

from sklearn.linear_model import LogisticRegressionCV
from torch.autograd import Variable
from utils.mahalanobis_lib import get_Mahalanobis_score
from utils.data_loader import get_dataloader_in, get_dataloader_out, cifar_out_datasets, imagenet_out_datasets
from utils.model_loader import get_model

from utils.cal_score import *
from argparser import *

def run_eval(model, in_loader, out_loader, logger, args, num_classes, out_dataset, mask=None, class_mean=None):
    # switch to evaluate mode
    model.eval()

    logger.info("Running test...")
    logger.flush()
    if args.score == 'HIMPLoS':
        logger.info("Processing in-distribution data...")
        in_scores = iterate_data_HIMPLoS(in_loader, model, args.temperature_energy, mask, args.p, args.threshold, class_mean)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_HIMPLoS(out_loader, model, args.temperature_energy, mask, args.p, args.threshold, class_mean)
    elif args.score == 'MSP':
        logger.info("Processing in-distribution data...")
        in_scores = iterate_data_msp(in_loader, model)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_msp(out_loader, model)
    elif args.score == 'ODIN':
        logger.info("Processing in-distribution data...")
        in_scores = iterate_data_odin(in_loader, model, args.epsilon_odin, args.temperature_odin, logger)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_odin(out_loader, model, args.epsilon_odin, args.temperature_odin, logger)
    elif args.score == 'Energy':
        logger.info("Processing in-distribution data...")
        in_scores = iterate_data_energy(in_loader, model, args.temperature_energy)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_energy(out_loader, model, args.temperature_energy)
    elif args.score == 'dice':
        info = np.load(f"checkpoints/feature/{args.name}/{args.in_dataset}/{args.model}_feat_stat.npy")
        model = get_model(args, num_classes, load_ckpt=True, info=info)
        model.eval()
        logger.info("Processing in-distribution data...")
        in_scores = iterate_data_energy(in_loader, model, args.temperature_energy)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_energy(out_loader, model, args.temperature_energy)
    elif args.score == 'dice_react':
        info = np.load(f"checkpoints/feature/{args.name}/{args.in_dataset}/{args.model}_feat_stat.npy")
        model = get_model(args, num_classes, load_ckpt=True, info=info)
        model.eval()
        logger.info("Processing in-distribution data...")
        in_scores = iterate_data_react(in_loader, model, args.temperature_energy, args.threshold)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_react(out_loader, model, args.temperature_energy, args.threshold)
    elif args.score == 'react':
        logger.info("Processing in-distribution data...")
        in_scores = iterate_data_react(in_loader, model, args.temperature_energy, args.threshold)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_react(out_loader, model, args.temperature_energy, args.threshold)
    elif args.score == 'LINE':
        info = np.load(f"cache/{args.name}/{args.in_dataset}_{args.model}_meanshap_class.npy")
        model = get_model(args, num_classes, load_ckpt=True, info=info, LU=True)
        model.eval()
        logger.info("Processing in-distribution data...")
        in_scores = iterate_data_LINE(in_loader, model, args.temperature_energy, args.threshold)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_LINE(out_loader, model, args.temperature_energy, args.threshold)
    elif args.score == 'Mahalanobis':
        save_dir = os.path.join('cache', 'mahalanobis', args.name)
        lr_weights, lr_bias, magnitude = np.load(
            os.path.join(save_dir, f'{args.in_dataset}_{args.model}_results.npy'), allow_pickle=True)

        regressor = LogisticRegressionCV(cv=2).fit([[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]],
                                                   [0, 0, 1, 1])


        regressor.coef_ = lr_weights
        regressor.intercept_ = lr_bias

        temp_x = torch.rand(2, 3, 32, 32)
        temp_x = Variable(temp_x).cuda()
        # temp_list = model(x=temp_x, layer_index='all')[1]
        temp_list = model.feature_list(temp_x)[1]
        num_output = len(temp_list)

        file_folder = os.path.join('cache', 'mahalanobis', args.name)
        filename1 = os.path.join(file_folder, f'{args.in_dataset}_{args.model}_class_mean.npy')
        filename2 = os.path.join(file_folder, f'{args.in_dataset}_{args.model}_precision.npy')
        sample_mean = np.load(filename1, allow_pickle=True)
        precision = np.load(filename2, allow_pickle=True)

        sample_mean = [s.cuda() for s in sample_mean]
        precision = [torch.from_numpy(p).float().cuda() for p in precision]


        logger.info("Processing in-distribution data...")
        in_scores = iterate_data_mahalanobis(in_loader, model, num_classes, sample_mean, precision,
                                             num_output, magnitude, regressor, logger)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_mahalanobis(out_loader, model, num_classes, sample_mean, precision,
                                              num_output, magnitude, regressor, logger)
    
    else:
        raise ValueError("Unknown score type {}".format(args.score))

    in_examples = in_scores.reshape((-1, 1))
    out_examples = out_scores.reshape((-1, 1))

    auroc, aupr_in, aupr_out, fpr95 = get_measures(in_examples, out_examples)
    auroc, aupr_in, aupr_out, fpr95 = auroc*100, aupr_in*100, aupr_out*100, fpr95*100
    
    logger.info('============Results for {}============'.format(args.score))
    logger.info('=======in dataset: {}; ood dataset: {}============'.format(args.in_dataset, out_dataset))
    logger.info('AUROC: {}'.format(auroc))
    logger.info('AUPR (In): {}'.format(aupr_in))
    logger.info('AUPR (Out): {}'.format(aupr_out))
    logger.info('FPR95: {}'.format(fpr95))

    logger.flush()
    return auroc, aupr_in, aupr_out, fpr95


def extact_mean_std(args, model):
    for key, v in model.state_dict().items():
        if key == 'classifier.1.weight':
            fc_w = v
        if key == 'fc.weight':
            fc_w = v
        
    return fc_w.cpu().numpy()

def get_mask_classmean(args, fc_w):
    file_folder = f'checkpoints/feature/{args.name}/{args.in_dataset}'
    class_mean = np.load(f"{file_folder}/{args.model}_class_mean.npy")

    p = 0
    if args.p:
        p = args.p

    thresh = np.percentile(fc_w, p, axis=1)
    mask = np.zeros_like(fc_w)
    for i in range(mask.shape[0]):
        mask[i] = np.where(fc_w[i] >= thresh[i],1,0)

    mask = torch.tensor(mask)
    class_mean = torch.tensor(class_mean)
    return mask, class_mean


def main(args):
    logger = log.setup_logger(args)

    in_dataset = args.in_dataset

    in_save_dir = os.path.join(args.logdir, args.name, args.model)
    if not os.path.exists(in_save_dir):
        os.makedirs(in_save_dir)

    loader_in_dict = get_dataloader_in(args, split=('val'))
    in_loader, num_classes = loader_in_dict.val_loader, loader_in_dict.num_classes
    args.num_classes = num_classes

    load_ckpt = False
    if args.model_path != None:
        load_ckpt = True

    model = get_model(args, num_classes, load_ckpt=load_ckpt)
    
    model.eval()
    fc_w = extact_mean_std(args, model)
    mask, class_mean = get_mask_classmean(args, fc_w)
    class_mean = class_mean.cuda()

    if args.out_dataset is not None:
        out_dataset = args.out_dataset
        loader_out_dict = get_dataloader_out(args, (None, out_dataset), split=('val'))
        out_loader = loader_out_dict.val_ood_loader

        in_set, out_set = loader_in_dict.val_dataset, loader_out_dict.val_dataset
        logger.info(f"Using an in-distribution set with {len(in_set)} images.")
        logger.info(f"Using an out-of-distribution set with {len(out_set)} images.")


        start_time = time.time()
        run_eval(model, in_loader, out_loader, logger, args, num_classes=num_classes, out_dataset=out_dataset,  mask=mask, class_mean=class_mean)
        end_time = time.time()

        logger.info("Total running time: {}".format(end_time - start_time))
    
    else:
        out_datasets = []
        AUroc, AUPR_in, AUPR_out, Fpr95 = [], [], [], []
        if in_dataset == "imagenet":
            out_datasets = imagenet_out_datasets
        else:
            out_datasets = cifar_out_datasets
        for out_dataset in out_datasets:
            loader_out_dict = get_dataloader_out(args, (None, out_dataset), split=('val'))
            out_loader = loader_out_dict.val_ood_loader

            in_set, out_set = loader_in_dict.val_dataset, loader_out_dict.val_dataset
            logger.info(f"Using an in-distribution set with {len(in_set)} images.")
            logger.info(f"Using an out-of-distribution set with {len(out_set)} images.")

            start_time = time.time()
            auroc, aupr_in, aupr_out, fpr95 = run_eval(model, in_loader, out_loader, logger, args, num_classes=num_classes, out_dataset=out_dataset,  mask=mask, class_mean=class_mean)
            end_time = time.time()

            logger.info("Total running time: {}".format(end_time - start_time))

            AUroc.append(auroc)
            AUPR_in.append(aupr_in)
            AUPR_out.append(aupr_out)
            Fpr95.append(fpr95)
        avg_auroc = sum(AUroc) / len(AUroc)
        avg_aupr_in = sum(AUPR_in) / len(AUPR_in)
        avg_aupr_out = sum(AUPR_out) / len(AUPR_out)
        avg_fpr95 = sum(Fpr95) / len(Fpr95)



        logger.info('============Results for {}============'.format(args.score))
        logger.info('=======in dataset: {}; ood dataset: Average============'.format(args.in_dataset))
        logger.info('Average AUROC: {}'.format(avg_auroc))
        logger.info('Average AUPR (In): {}'.format(avg_aupr_in))
        logger.info('Average AUPR (Out): {}'.format(avg_aupr_out))
        logger.info('Average FPR95: {}'.format(avg_fpr95))



if __name__ == "__main__":
    parser = get_argparser()

    args = parser.parse_args()

    
    main(args)
