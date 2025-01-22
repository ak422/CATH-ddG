import os
import random
import sys
import time
import shutil
import re
import argparse
from colorama import Fore
from sklearn import preprocessing
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm
from torchsummary import summary
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from DDAffinity.utils.misc import BlackHole, load_config, seed_all, get_logger, get_new_dir, current_milli_time
from DDAffinity.models.DDAffinity import DDAffinity_NET
from DDAffinity.utils.skempi_mpnn import SkempiDatasetManager, eval_skempi_three_modes, eval_perprotein_modes
from DDAffinity.utils.transforms import get_transform
from DDAffinity.utils.train_mpnn import *
from DDAffinity.utils.early_stopping import EarlyStopping

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--logdir', type=str, default='./logs_skempi')
    parser.add_argument('--early_stoppingdir', type=str, default='./early_stopping')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--tag', type=str, default='')
    args = parser.parse_args()

    # Load configs
    config, config_name = load_config(args.config)
    seed_all(config.train.seed)

    # Logging
    if args.debug:
        logger = get_logger('train', None)
        writer = BlackHole()
        ckpt_dir = None
    else:
        log_dir = get_new_dir(args.logdir, prefix='[%d-fold]' % (config.train.num_cvfolds), tag=args.tag)
        early_stoppingdir = get_new_dir(args.early_stoppingdir, prefix='[%d-fold-%d]' % (config.train.num_cvfolds,config.model.k1), tag=args.tag)

        ckpt_dir = os.path.join(log_dir, 'checkpoints')
        if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
        logger = get_logger('train', log_dir)
        writer = torch.utils.tensorboard.SummaryWriter(log_dir)
        tensorboard_trace_handler = torch.profiler.tensorboard_trace_handler(log_dir)
        if not os.path.exists(os.path.join(log_dir, os.path.basename(args.config))):
            shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    logger.info(args)
    logger.info(config)

    # Data
    logger.info('Loading datasets...')
    dataset_mgr = SkempiDatasetManager(
        config,
        split_seed=config.train.seed,
        num_cvfolds=config.train.num_cvfolds,
        num_workers=args.num_workers,
        logger=logger,
    )

    # Model, Optimizer & Scheduler
    logger.info('Building model...')
    cv_mgr = CrossValidation(
        model_factory=DDAffinity_NET,
        config=config,
        early_stoppingdir=early_stoppingdir,
        num_cvfolds=config.train.num_cvfolds,
        logger=logger,
    ).to(args.device)
    it_first = 1  # epoch from 1 for warmup_CosineAnneal

    def train_one_epoch(fold, epoch):
        model, optimizer, scheduler, early_stopping = cv_mgr.get(fold)
        if early_stopping.early_stop == True:
            return fold

        time_start = current_milli_time()
        mean_loss = torch.zeros(1).to(args.device)
        model.train()
        # Prepare data
        train_loader = dataset_mgr.get_train_loader(fold)
        train_loader = tqdm(train_loader, file=sys.stdout, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.WHITE, Fore.RESET))

        for step, data in enumerate(train_loader):
            batch = recursive_to(data, args.device)
            # Forward pass
            loss_dict = model(batch)
            loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
            time_forward_end = current_milli_time()

            optimizer.zero_grad()  # 保证梯度为0

            # Backward
            loss.backward()

            mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses
            train_loader.desc = f"\033[0;37;42m [epoch {epoch}/{config.train.max_epochs} fold {fold+1}/{config.train.num_cvfolds}] mean loss {round(mean_loss.item(), 4)}\033[0m"

            orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

            if config.train.optimizer.type == 'adam':
                scheduler.step()

        time_backward_end = current_milli_time()
        logger.info(f'[epoch {epoch}/{config.train.max_epochs} fold {fold+1}/{config.train.num_cvfolds}] mean loss {mean_loss.item():.4f}')

        if epoch >= config.train.early_stopping_epoch  and early_stopping.early_stop == False:
            early_stopping(mean_loss.item(), model, fold)

    def validate(epoch):
        scalar_accum = ScalarMetricAccumulator()
        results = []

        with torch.no_grad():
            for fold in range(config.train.num_cvfolds):
                model, optimizer, scheduler, early_stopping = cv_mgr.get(fold)
                model.eval()
                for i, batch in enumerate(tqdm(dataset_mgr.get_val_loader(fold), desc=f'\033[0;37;42m Testing fold {fold+1}/{config.train.num_cvfolds}\033[0m', dynamic_ncols=True, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.WHITE, Fore.RESET))):
                    # Prepare data
                    batch = recursive_to(batch, args.device)

                    # Forward pass
                    output_dict = model.inference(batch)

                    for pdbcode, protein_group, complex_PPI, mutstr, ddg_true, ddg_pred in zip(batch["wt"]['#Pdb'], batch["wt"]['protein_group'], batch["wt"]['complex_PPI'], batch["wt"]['mutstr'], output_dict['ddG_true'], output_dict['ddG_pred']):
                        results.append({
                            'pdbcode': pdbcode,
                            'protein_group': protein_group,
                            'complex_PPI': complex_PPI,
                            'mutstr': mutstr,
                            'num_muts': len(mutstr.split(',')),
                            'ddG': ddg_true.item(),
                            'ddG_pred': ddg_pred.item(),
                        })

        results = pd.DataFrame(results)
        # PDB:1E96.pdb and 1E50.pdb
        results.replace("1.00E+96", "1E96", inplace=True)
        results.replace("1.00E+50", "1E50", inplace=True)

        results = results.groupby(['pdbcode', 'complex_PPI', 'protein_group']).agg(ddG_pred_mean=("ddG_pred", "mean"),
                                                                              ddG=("ddG", "mean"),
                                                                              num_muts=("num_muts", "mean")).reset_index()
        results['ddG_pred'] = results['ddG_pred_mean']
        results['datasets'] = 'SKEMPI2'

        if ckpt_dir is not None:
            results.to_csv(os.path.join(ckpt_dir, f'results_{epoch}.csv'), index=False)

        # 显示所有列,保留4位小数
        pd.set_option('display.max_columns', None)
        pd.options.display.float_format = '{:.4f}'.format

        df_metrics = eval_skempi_three_modes(results)
        pearson_score = df_metrics['overall_pearson'][2]
        df_metrics_string = df_metrics.to_string()
        logger.info('Our results:\n%s', df_metrics_string)

        if config.data.PPIformer == True:
            perprotein_metrics = eval_perprotein_modes(results)
            perprotein_metrics_string = perprotein_metrics.to_string()
            logger.info('PPIformer results:\n%s', perprotein_metrics_string)

        return pearson_score

    try:
        logger.info('Training model ...')
        early_stopping_sets = set()
        for epoch in range(it_first, config.train.max_epochs + 1):
            # early stopping
            if len(early_stopping_sets) > config.train.num_cvfolds:     # count None
                break
            for fold in range(config.train.num_cvfolds):
                fold_flags = train_one_epoch(fold, epoch)
                early_stopping_sets.add(fold_flags)

            if (config.train.num_cvfolds == 1 and config.data.cath_fold == False and config.data.PPIformer == False):
                continue

            if epoch % config.train.val_freq == 0:
                pearson_score = validate(epoch)

                if epoch >= 150:
                    date_time = re.search(r'\d{2}_\d{2}_\d{2}_\d{2}_\d{2}', log_dir)
                    ckpt_path = os.path.join(ckpt_dir, '%s_epoch%d.pt' % (date_time.group(),epoch))
                    torch.save({
                        'config': config,
                        'model': cv_mgr.state_dict(),
                        'epoch': epoch,
                    }, ckpt_path)

        if True:
            # Saving 3-fold checkpoint: DDAffinity
            date_time = re.search(r'\d{2}_\d{2}_\d{2}_\d{2}_\d{2}', early_stoppingdir)
            checkpoint = '%s_DDAffinity.pt' % date_time.group()
            logger.info(f'Saving {config.train.num_cvfolds}-fold best_network: {checkpoint}')
            cv_mgr.save_state_dict(args, config, early_stoppingdir, checkpoint)
            # Loading checkpoint: DDAffinity
            ckpt_path = os.path.join(early_stoppingdir, checkpoint)
            ckpt = torch.load(ckpt_path, map_location=args.device)
            cv_mgr.load_state_dict(ckpt['model'], )

            if (config.train.num_cvfolds == 1 and config.data.cath_fold == False and config.data.PPIformer == False):
                exit()

            results = []
            with torch.no_grad():
                for fold in range(config.train.num_cvfolds):
                    logger.info(f'Loading from checkpoint: Fold_{fold+1}_best_network.pt')
                    model, _, _, _ = cv_mgr.get(fold)
                    model.eval()
                    for i, batch in enumerate(tqdm(dataset_mgr.get_val_loader(fold), desc='Validate', dynamic_ncols=True, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.WHITE, Fore.RESET))):
                        # Prepare data
                        batch = recursive_to(batch, args.device)
                        # Forward pass
                        output_dict = model.inference(batch)
                        for pdbcode, protein_group, complex_PPI, mutstr, ddg_true, ddg_pred in zip(batch["wt"]['#Pdb'], batch["wt"]['protein_group'], batch["wt"]['complex_PPI'], batch["wt"]['mutstr'],output_dict['ddG_true'],output_dict['ddG_pred']):
                            results.append({
                                'pdbcode': pdbcode,
                                'protein_group': protein_group,
                                'complex_PPI': complex_PPI,
                                'mutstr': mutstr,
                                'num_muts': len(mutstr.split(',')),
                                'ddG': ddg_true.item(),
                                'ddG_pred': ddg_pred.item(),
                            })

            results = pd.DataFrame(results)
            # PDB:1E96.pdb and 1E50.pdb
            results.replace("1.00E+96", "1E96", inplace=True)
            results.replace("1.00E+50", "1E50", inplace=True)

            results = results.groupby(['pdbcode', 'complex_PPI', 'protein_group']).agg(ddG_pred_mean=("ddG_pred", "mean"),
                                                                                  ddG=("ddG", "mean"),
                                                                                  num_muts=("num_muts", "mean")).reset_index()
            results['ddG_pred'] = results['ddG_pred_mean']
            results['datasets'] = 'SKEMPI2'

            if ckpt_dir is not None:
                results.to_csv(os.path.join(ckpt_dir, f'results_final.csv'), index=False)

            # 显示所有列,保留4位小数
            pd.set_option('display.max_columns', None)
            pd.options.display.float_format = '{:.4f}'.format

            df_metrics = eval_skempi_three_modes(results)
            df_metrics_string = df_metrics.to_string()
            logger.info('Our results:\n%s', df_metrics_string)

            if config.data.PPIformer == True:
                perprotein_metrics = eval_perprotein_modes(results)
                perprotein_metrics_string = perprotein_metrics.to_string()
                logger.info('PPIformer results:\n%s', perprotein_metrics_string)

    except KeyboardInterrupt:
        logger.info('Terminating...')
