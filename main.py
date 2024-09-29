import sys
import copy
import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import to_hetero

import utils
from utils import create_output_dir, get_path, gaussian_normalize
from model import HiUrNet
from args import args
from graph import HeterogeneousODGraph
from loss import choose_criterion_type
from preprocessing import preprocessing


def train(model, data, optimizer, scheduler, criterion, clip_threshold, multi_task_weight):
    model.train()
    optimizer.zero_grad()
    edge_label_index_dict = {'m2c': data['mesh', 'm2c', 'city'].edge_label_index,
                             'c2m': data['city', 'c2m', 'mesh'].edge_label_index,
                             'm2m': data['mesh', 'm2m', 'mesh'].edge_label_index}
    target = {'c2m': data['city', 'c2m', 'mesh'].edge_label, 'm2c': data['mesh', 'm2c', 'city'].edge_label,
              'm2m': data['mesh', 'm2m', 'mesh'].edge_label}
    out = {}

    out['c2m'], out['m2c'], out['m2m'] = model(data.x_dict,
                                               data.edge_index_dict,
                                               edge_label_index_dict)
    c2m_loss = criterion(out['c2m'], target['c2m'])
    m2c_loss = criterion(out['m2c'], target['m2c'])
    m2m_loss = criterion(out['m2m'], target['m2m'])
    loss = multi_task_weight[0] * c2m_loss + multi_task_weight[1] * m2c_loss + multi_task_weight[2] * m2m_loss
    mse = {'c2m': F.mse_loss(out['c2m'], target['c2m']).sqrt(),
           'm2c': F.mse_loss(out['m2c'], target['m2c']).sqrt(),
           'm2m': F.mse_loss(out['m2m'], target['m2m']).sqrt()}
    loss.backward()
    if clip_threshold is not None:
        nn.utils.clip_grad_norm_(model.parameters(), clip_threshold)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return float(loss), mse, float(c2m_loss), float(m2c_loss), float(m2m_loss)


@torch.no_grad()
def test(model, data, criterion, multi_task_weight):
    model.eval()
    edge_label_index_dict = {'m2c': data['mesh', 'm2c', 'city'].edge_label_index,
                             'c2m': data['city', 'c2m', 'mesh'].edge_label_index,
                             'm2m': data['mesh', 'm2m', 'mesh'].edge_label_index}
    target = {'c2m': data['city', 'c2m', 'mesh'].edge_label, 'm2c': data['mesh', 'm2c', 'city'].edge_label,
              'm2m': data['mesh', 'm2m', 'mesh'].edge_label}
    out = {}
    r_2 = {}

    out['c2m'], out['m2c'], out['m2m'] = model(data.x_dict,
                                               data.edge_index_dict,
                                               edge_label_index_dict)

    c2m_loss = criterion(out['c2m'], target['c2m'])
    m2c_loss = criterion(out['m2c'], target['m2c'])
    m2m_loss = criterion(out['m2m'], target['m2m'])
    loss = multi_task_weight[0] * c2m_loss + multi_task_weight[1] * m2c_loss + multi_task_weight[2] * m2m_loss
    out['c2m'][out['c2m'] < 0] = 0
    out['m2c'][out['m2c'] < 0] = 0
    out['m2m'][out['m2m'] < 0] = 0
    r_2['c2m'] = np.corrcoef(out['c2m'].squeeze().cpu().detach().numpy(),
                             target['c2m'].squeeze().cpu().detach().numpy())[0][1] ** 2
    r_2['m2c'] = np.corrcoef(out['m2c'].squeeze().cpu().detach().numpy(),
                             target['m2c'].squeeze().cpu().detach().numpy())[0][1] ** 2
    r_2['m2m'] = np.corrcoef(out['m2m'].squeeze().cpu().detach().numpy(),
                             target['m2m'].squeeze().cpu().detach().numpy())[0][1] ** 2
    return float(loss), out, target, r_2


def model_hetero(result_folder):
    utils.save_dict_to_json(vars(args), os.path.join(result_folder, 'param.json'))
    logger = logging.getLogger(__name__)
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = torch.device(args.device)

    utils.seed_everything(args.seed)

    logger.info('Device: {}'.format(device))

    multi_task_weight = [(1 - args.multi_task_weight_m2m) / 2,
                         (1 - args.multi_task_weight_m2m) / 2,
                         args.multi_task_weight_m2m]

    dict_df_data = preprocessing()

    if args.layer_type == 'GAT':
        g = HeterogeneousODGraph(dict_df_data['xs'], dict_df_data['edges']['od'], dict_df_data['edges']['include'])
    else:
        g = HeterogeneousODGraph(dict_df_data['xs'], dict_df_data['edges']['od'], dict_df_data['edges']['include'],
                                 neighbor=dict_df_data['edges']['neighbor'])
    data = g.graph

    data['mesh'].x = gaussian_normalize(data['mesh'].x)

    early_stopper = utils.EarlyStopper(patience=args.early_stopper_patience, delta=args.early_stopper_delta)

    transform = T.Compose([T.ToDevice(device),
                           T.RandomLinkSplit(num_val=0.05,
                                             num_test=0.1,
                                             neg_sampling_ratio=0.0,
                                             edge_types=[('mesh', 'm2c', 'city'),
                                                         ('city', 'c2m', 'mesh'),
                                                         ('mesh', 'm2m', 'mesh')])])

    assert data['mesh'].x.shape[1] == 43 or 86
    assert data['city'].x.shape[0] == 43
    assert data['city'].x.shape[1] == 43 or 86

    train_data, val_data, test_data = transform(data)

    # Delete edge_index for disabling message passing
    if args.layer_type == 'HGT':
        if not args.flow:
            del train_data['c2m'].edge_index
            del train_data['m2c'].edge_index
            del train_data['m2m'].edge_index
            del val_data['c2m'].edge_index
            del val_data['m2c'].edge_index
            del val_data['m2m'].edge_index
            del test_data['c2m'].edge_index
            del test_data['m2c'].edge_index
            del test_data['m2m'].edge_index

        if not args.inclusion:
            del train_data['isin'].edge_index
            del train_data['include'].edge_index
            del val_data['isin'].edge_index
            del val_data['include'].edge_index
            del test_data['isin'].edge_index
            del test_data['include'].edge_index

        if args.geo:
            pass
        else:
            del train_data['near'].edge_index
            del val_data['near'].edge_index
            del test_data['near'].edge_index

    logger.info('The summary of the message passing edge: ')
    for k, v in train_data.edge_index_dict.items():
        logger.info('The edge type {} has the size {}: '.format(k, v.shape))

    g.convert_split_edge_index_with_edge_value(train_data).to_csv(
        os.path.join(result_folder, 'train_edges.csv'), header=False, index=False)
    g.convert_split_edge_index_with_edge_value(val_data).to_csv(
        os.path.join(result_folder, 'val_edges.csv'), header=False, index=False)
    g.convert_split_edge_index_with_edge_value(test_data).to_csv(
        os.path.join(result_folder, 'test_edges.csv'), header=False, index=False)

    model = HiUrNet(in_channels=data['mesh'].x.shape[1],
                    hidden_channels=args.hidden_channels,
                    num_city=data['city'].x.shape[0],
                    data=train_data,
                    num_layers=args.num_layer,
                    heads=args.heads,
                    dropout=args.dropout,
                    layer_type=args.layer_type)

    if args.layer_type == 'GAT':
        model.encoder = to_hetero(model.encoder, data.metadata(), aggr='sum')

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.lr_weight_decay)

    if args.scheduled:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=args.scheduler_step,
                                                    gamma=args.scheduler_gamma)
    else:
        scheduler = None

    # Lazy initialization: run one model step to infer the number of parameters
    with torch.no_grad():
        model.encoder(train_data.x_dict, train_data.edge_index_dict)

    best_loss = float('inf')
    best_model = None
    best_epoch = 1

    train_losses = []
    val_losses = []
    r_2s = {'c2m': [], 'm2c': [], 'm2m': []}

    criterion = choose_criterion_type(args.loss_type)

    # Start training
    start = time.time()
    for epoch in range(1, args.epochs):
        train_loss, _, c2m_loss, m2c_loss, m2m_loss = train(model,
                                                            train_data,
                                                            optimizer,
                                                            scheduler,
                                                            criterion,
                                                            args.clip_threshold,
                                                            multi_task_weight)
        val_loss, _, _, r_2 = test(model, val_data, criterion, multi_task_weight)
        if epoch % 10 == 1:
            logger.info('Epoch: {:07d}, Training Loss: {:.4f}, Validation Loss: {:.4f}, '
                        '(c2m, m2c, m2m): {:.4f}, {:.4f}, {:.4f}'.format(epoch,
                                                                         train_loss, val_loss,
                                                                         c2m_loss, m2c_loss, m2m_loss))
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        r_2s['c2m'].append(r_2['c2m'])
        r_2s['m2c'].append(r_2['m2c'])
        r_2s['m2m'].append(r_2['m2m'])

        if epoch % 10 == 1:
            state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optim_dict': optimizer.state_dict()}
            utils.save_checkpoint(state, is_best=val_loss < best_loss, folder=result_folder)

        if best_loss > val_loss:
            best_epoch = epoch
            best_loss = val_loss
            best_model = copy.deepcopy(model)

        early_stopper(val_loss, model)
        if early_stopper.early_stop:
            logger.info('Early Stopping!')
            break
    # End training
    end = time.time()
    logger.info('Training time elapsed: {}'.format(end - start))
    logger.info('Best epoch is: {}'.format(best_epoch))

    # Start inferring
    start = time.time()
    test_loss, out, target, _ = test(best_model, test_data, criterion, multi_task_weight)
    # End inferring
    end = time.time()
    logger.info('Inferring time elapsed: {}'.format(end - start))

    g.convert_split_edge_index_with_edge_value(test_data, [out, target]).to_csv(
        os.path.join(result_folder, 'test_prediction.csv'), header=False, index=False)

    result_c2m = utils.Metrics(out['c2m'], target['c2m'])
    result_m2c = utils.Metrics(out['m2c'], target['m2c'])
    result_m2m = utils.Metrics(out['m2m'], target['m2m'])

    logger.info('C2M L2 Loss: {}'.format(result_c2m.rmse() ** 2))
    logger.info('C2M RMSE: {}'.format(result_c2m.rmse()))
    logger.info('C2M L1 Loss: {}'.format(result_c2m.mae()))
    logger.info('C2M PCC: {}'.format(result_c2m.pcc()))
    logger.info('C2M R_2: {}'.format(result_c2m.r_2()))
    logger.info('C2M cpc: {}'.format(result_c2m.cpc()))
    logger.info('M2C L2 Loss: {}'.format(result_m2c.rmse() ** 2))
    logger.info('M2C RMSE: {}'.format(result_m2c.rmse()))
    logger.info('M2C L1 Loss: {}'.format(result_m2c.mae()))
    logger.info('M2C PCC: {}'.format(result_m2c.pcc()))
    logger.info('M2C R_2: {}'.format(result_m2c.r_2()))
    logger.info('M2C cpc: {}'.format(result_m2c.cpc()))
    logger.info('M2M L2 Loss: {}'.format(result_m2m.rmse() ** 2))
    logger.info('M2M RMSE: {}'.format(result_m2m.rmse()))
    logger.info('M2M L1 Loss: {}'.format(result_m2m.mae()))
    logger.info('M2M PCC: {}'.format(result_m2m.pcc()))
    logger.info('M2M R_2: {}'.format(result_m2m.r_2()))
    logger.info('M2M cpc: {}'.format(result_m2m.cpc()))

    logger.info('Test Loss:{:.4f}'.format(test_loss))


def main():
    result_folder = create_output_dir(args.save_folder, args.experiment_name)
    # set output logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(get_path(os.path.join(result_folder, 'log.txt')))
    console_handler = logging.StreamHandler(sys.stdout)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    targets = console_handler, file_handler
    logger.handlers = targets

    model_hetero(result_folder)


if __name__ == '__main__':
    main()
