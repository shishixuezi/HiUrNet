import os
import torch
import numpy as np
import pandas as pd
import torch_geometric
import torch_geometric.transforms as T
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from torch_geometric.explain.config import ExplanationType

from args import args
from torch_geometric.nn import to_hetero
from torch_geometric.explain import CaptumExplainer, Explainer, HeteroExplanation
from graph import HeterogeneousODGraph
from model import HiUrNet
from preprocessing import preprocessing
from utils import load_checkpoint

from typing import Optional, Union, List


def hetero_explain(model,
                   g,
                   target_data,
                   explain_edge_type: Optional[Union[str, List[str]]] = None,
                   name='',
                   explain_edge_list=None):

    model.eval()

    explainer = Explainer(
        model=model,
        algorithm=CaptumExplainer('IntegratedGradients'),
        explanation_type='model',
        model_config=dict(mode="regression", task_level="edge", return_type="raw"),
        node_mask_type="attributes",
        edge_mask_type="object",
        threshold_config=dict(threshold_type='topk', value=args.explain_threshold)
    )

    if explain_edge_list is None:
        # The index of the edge_label
        index = torch.tensor([0, 1])
    else:
        index = torch.tensor(explain_edge_list[:200])

    edge_label_index_dict = {'m2c': target_data['mesh', 'm2c', 'city'].edge_label_index,
                             'c2m': target_data['city', 'c2m', 'mesh'].edge_label_index,
                             'm2m': target_data['mesh', 'm2m', 'mesh'].edge_label_index}

    # call the explainer to generate an explanation
    explanation = explainer(
        target_data.x_dict,
        target_data.edge_index_dict,
        index=index,
        edge_label_index_dict=edge_label_index_dict,
        target_type=explain_edge_type
    )

    assert 'node_mask' in explanation.available_explanations
    assert 'edge_mask' in explanation.available_explanations
    assert isinstance(explanation, HeteroExplanation)
    # assert model.training
    if explain_edge_type == ExplanationType.phenomenon:
        for key in explanation.num_types:
            assert explanation[key].node_mask.size() == target_data[key].x.size()
        for key in explanation.available_explanations:
            print(f'Generated explanation in {key}')
            for mask in explanation.collect(key).values():
                assert set(mask.unique().tolist()).issubset({0, 1})

    sub_graph = explanation.get_explanation_subgraph()

    # get feature masks
    node_mask_result = {}
    node_score_dict = {'mesh': explanation.node_mask_dict['mesh'].squeeze(0).abs().sum(dim=0),
                       'city': explanation.node_mask_dict['city'].squeeze(0).abs().sum(dim=0)}
    for node in node_score_dict.keys():
        node_score_dict[node] /= node_score_dict[node].max()
        df = pd.DataFrame(node_score_dict[node].cpu().detach().numpy())
        df.to_csv(os.path.join(args.explain_path, 'feature_importance_' + node + name + '.csv'), index=False)
        node_mask_result[node] = df

    # get node masks
    node_id_dict = {'mesh': explanation.node_mask_dict['mesh'].squeeze(0).abs().sum(dim=1).nonzero().squeeze(0),
                    'city': explanation.node_mask_dict['city'].squeeze(0).abs().sum(dim=1).nonzero().squeeze(0)}
    for node in node_id_dict.keys():
        p1 = torch.squeeze(node_id_dict[node]).cpu().detach().numpy()
        # get all features separately of all nodes
        p2 = sub_graph[node]['node_mask'].cpu().detach().numpy()
        result = pd.DataFrame(p2, columns=np.arange(p2.shape[1]))
        result.insert(0, 'id', p1, True)
        result.insert(1, 'KEY_CODE', result['id'].map({v: k for k, v in g.nodes[node].items()}), True)
        result.to_csv(os.path.join(args.explain_path, node + '_all_features_mask' + name + '.csv'), index=False)

        # sum all features of one specific node to check the importance
        p2 = sub_graph[node]['node_mask'].abs().sum(dim=1)
        p2 /= p2.max()
        p2 = p2.cpu().detach().numpy()
        result = pd.DataFrame({'node_id': p1, 'node_mask': p2})
        result['KEY_CODE'] = result['node_id'].map({v: k for k, v in g.nodes[node].items()})
        result.to_csv(os.path.join(args.explain_path, node+'_mask' + name + '.csv'), index=False)

    # get edge masks
    node_abbr_dict = {'c': 'city', 'm': 'mesh'}
    for edge in explain_edge_type:
        result = pd.DataFrame({'o_id': sub_graph[edge]['edge_index'][0].cpu().detach().numpy(),
                               'd_id': sub_graph[edge]['edge_index'][1].cpu().detach().numpy(),
                               'edge_mask': sub_graph[edge]['edge_mask'].cpu().detach().numpy()})

        result['o_KEY_CODE'] = result['o_id'].map({v: k for k, v in g.nodes[node_abbr_dict[edge[0]]].items()})
        result['d_KEY_CODE'] = result['d_id'].map({v: k for k, v in g.nodes[node_abbr_dict[edge[2]]].items()})
        result.to_csv(os.path.join(args.explain_path, edge+'_mask' + name + '.csv'), index=False)

    return node_mask_result['mesh'], node_mask_result['city']


def generate_het_explanation():
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    device = torch.device(device)

    dict_df_data = preprocessing()
    g = HeterogeneousODGraph(dict_df_data['xs'],
                             dict_df_data['edges']['od'],
                             dict_df_data['edges']['include'],
                             neighbor=dict_df_data['edges']['neighbor'])

    data = g.graph.to(device)
    transform = T.Compose([T.ToDevice('cuda' if torch.cuda.is_available() else 'cpu'),
                           T.RandomLinkSplit(num_val=0.05,
                                             num_test=0.1,
                                             neg_sampling_ratio=0.0,
                                             edge_types=[('mesh', 'm2c', 'city'),
                                                         ('city', 'c2m', 'mesh'),
                                                         ('mesh', 'm2m', 'mesh')])])
    train_data, val_data, test_data = transform(data)
    model = HiUrNet(in_channels=data['mesh'].x.shape[1],
                    layer_type=args.layer_type,
                    hidden_channels=args.hidden_channels,
                    num_city=data['city'].x.shape[0],
                    data=train_data,
                    num_layers=args.num_layer,
                    heads=args.heads,
                    dropout=args.dropout)

    if args.layer_type == 'GAT':
        model.encoder = to_hetero(model.encoder, data.metadata(), aggr='sum')

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.lr_weight_decay)
    path = os.path.join(args.explain_path, 'best.pth.tar')
    model_to_explain, _ = load_checkpoint(path, model, optimizer)

    c2m_mesh, c2m_city = hetero_explain(model_to_explain, g, test_data, explain_edge_type=['c2m'], name='_c2m')
    m2c_mesh, m2c_city = hetero_explain(model_to_explain, g, test_data, explain_edge_type=['m2c'], name='_m2c')

    sensitivity = pd.concat([c2m_mesh.T, c2m_city.T, m2c_mesh.T, m2c_city.T], ignore_index=True)
    visualize_het_explanation(sensitivity)


def visualize_het_explanation(sensitivity):
    color = ['green'] * 1 + ['blue'] * 17 + ['black'] * 24 + ['red'] * 1

    plt.style.use('ggplot')

    fig, axes = plt.subplots(2, 2, figsize=(16, 8))

    # ind = [0, 15, 32, 42]

    custom_lines = [Line2D([0], [0], color='green', lw=4),
                    Line2D([0], [0], color='blue', lw=4),
                    Line2D([0], [0], color='black', lw=4),
                    Line2D([0], [0], color='red', lw=4),
                    ]

    axes[0, 0].set_title('Grid in City-to-Grid Flows')
    axes[0, 1].set_title('City in City-to-Grid Flows')
    axes[1, 0].set_title('Grid in Grid-to-City Flows')
    axes[1, 1].set_title('City in Grid-to-City Flows')

    for ax in axes.flat:
        ax.set_ylabel('Feature Importance')
        ax.set_xticks([])
        ax.legend(custom_lines, ['Night Population', 'POIs', 'Road Density', 'Railway Users'])

    axes[0, 0].bar(range(sensitivity.shape[1]), sensitivity.iloc[0], color=color)
    axes[0, 1].bar(range(sensitivity.shape[1]), sensitivity.iloc[1], color=color)
    axes[1, 0].bar(range(sensitivity.shape[1]), sensitivity.iloc[2], color=color)
    axes[1, 1].bar(range(sensitivity.shape[1]), sensitivity.iloc[3], color=color)

    plt.savefig(os.path.join(args.explain_path, 'feature_analysis.pdf'), bbox_inches='tight')


if __name__ == '__main__':
    torch_geometric.seed_everything(args.seed)
    generate_het_explanation()
