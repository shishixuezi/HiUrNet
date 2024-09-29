from abc import ABC
from typing import Union, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn import Linear, GATv2Conv, HGTConv
from torch_geometric.typing import Metadata


class GATEncoder(nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers, heads=2, dropout=None):
        super().__init__()

        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.lins = nn.ModuleList()

        for _ in range(num_layers - 1):
            conv = GATv2Conv((-1, -1), hidden_channels, heads=heads, add_self_loops=False)
            self.convs.append(conv)
            lin = Linear(-1, hidden_channels * heads)
            self.lins.append(lin)
            norm = nn.BatchNorm1d(hidden_channels * heads)
            self.norms.append(norm)

        conv = GATv2Conv((-1, -1), hidden_channels, heads=heads, add_self_loops=False, concat=False)
        self.convs.append(conv)
        lin = Linear(-1, out_channels)
        self.lins.append(lin)
        norm = nn.BatchNorm1d(hidden_channels)
        self.norms.append(norm)

        self.use_dropout = True if dropout else False
        if self.use_dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x, edge_index):

        for i in range(self.num_layers):
            x = self.norms[i](self.convs[i](x, edge_index) + self.lins[i](x))
            if i < self.num_layers - 1:
                x = F.leaky_relu(x)
                x = self.dropout(x)
        return x


class HGTEncoder(nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers, data, heads=2, dropout=None):
        super().__init__()
        self.num_layers = num_layers
        self.lin_dict = nn.ModuleDict()
        self.preprocess_dict = nn.ModuleDict()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for node_type in data.node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)
            self.preprocess_dict[node_type] = MLP(-1, hidden_channels, hidden_channels, dropout=dropout, flatten=False)

        for _ in range(num_layers - 1):
            conv = MyGCN(hidden_channels, hidden_channels, data.metadata(), heads)
            self.convs.append(conv)
            norm = nn.LayerNorm(hidden_channels)
            self.norms.append(norm)

        conv = MyGCN(hidden_channels, out_channels, data.metadata(), heads)

        self.convs.append(conv)
        norm = nn.LayerNorm(hidden_channels)
        self.norms.append(norm)

        self.use_dropout = True if dropout else False
        if self.use_dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for k, v in self.lin_dict.items():
            v.reset_parameters()

    def forward(self, x_dict, edge_index_dict):
        x_dict = {
            node_type: F.gelu(self.lin_dict[node_type](x))
            for node_type, x in x_dict.items()
        }

        for i in range(self.num_layers):
            x_dict.update(self.convs[i](x_dict, edge_index_dict))
            x_dict = {key: self.dropout(self.norms[i](h)) for key, h in x_dict.items()}

        return x_dict


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=None, flatten=True):
        super().__init__()
        self.lin1 = Linear(in_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)
        self.use_dropout = True if dropout else False
        if self.use_dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = nn.Identity()
        self.flatten = flatten
        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x):
        z = F.leaky_relu(self.lin1(x))
        z = self.dropout(z)
        z = self.lin2(z)
        if self.flatten:
            return z.view(-1)
        else:
            return z


class HiUrNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_city, num_layers=3, data=None, heads=2, dropout=None,
                 layer_type='HGT'):
        super().__init__()
        self.city_embed = torch.nn.Embedding(num_city, in_channels)
        if layer_type == 'GAT':
            self.encoder = GATEncoder(hidden_channels, hidden_channels, num_layers, heads=2, dropout=dropout)
        elif layer_type == 'HGT':
            self.encoder = HGTEncoder(hidden_channels, hidden_channels, num_layers, data, heads=heads, dropout=dropout)
        else:
            raise NotImplementedError
        self.decoder_c2m = MLP(hidden_channels*2, hidden_channels, 1, dropout=dropout)
        self.decoder_m2c = MLP(hidden_channels*2, hidden_channels, 1, dropout=dropout)
        self.decoder_m2m = MLP(hidden_channels*2, hidden_channels, 1, dropout=dropout)

    def forward(self, x_dict, edge_index_dict, edge_label_index_dict, target_type=None):
        # if features are all zeros, use an embedding layer to learn feature distribution
        if not x_dict['city'].any():
            x_dict['city'] = self.city_embed(torch.arange(x_dict['city'].shape[0], device=x_dict['city'].get_device()))

        if 'pref' in x_dict:
            x_dict['pref'] = self.pref_embed(torch.arange(x_dict['pref'].shape[0],
                                                          device=x_dict['pref'].get_device_name()))
        if 'region' in x_dict:
            x_dict['region'] = self.region_embed(torch.arange(x_dict['region'].shape[0],
                                                              device=x_dict['region'].get_device_name()))

        x_dict.update({k: v for k, v in x_dict.items()})

        x_dict = self.encoder(x_dict, edge_index_dict)

        city_emb_c2m = x_dict['city'][edge_label_index_dict['c2m'][0]]
        mesh_emb_c2m = x_dict['mesh'][edge_label_index_dict['c2m'][1]]
        mesh_emb_m2c = x_dict['mesh'][edge_label_index_dict['m2c'][0]]
        city_emb_m2c = x_dict['city'][edge_label_index_dict['m2c'][1]]
        mesh1_emb_m2m = x_dict['mesh'][edge_label_index_dict['m2m'][0]]
        mesh2_emb_m2m = x_dict['mesh'][edge_label_index_dict['m2m'][1]]

        z_c2m = torch.cat([city_emb_c2m, mesh_emb_c2m], dim=-1)
        z_m2c = torch.cat([mesh_emb_m2c, city_emb_m2c], dim=-1)
        z_m2m = torch.cat([mesh1_emb_m2m, mesh2_emb_m2m], dim=-1)

        return_type_dict = {'c2m': z_c2m, 'm2m': z_m2m, 'm2c': z_m2c}
        decode_type_dict = {'c2m': self.decoder_c2m, 'm2m': self.decoder_m2m, 'm2c': self.decoder_m2c}
        if isinstance(target_type, list):
            # For explanation, if explaining multiple edge types, concatenate the predicted values
            r = []
            for item in target_type:
                assert not return_type_dict[item] is None
                r.append(decode_type_dict[item](return_type_dict[item]))
            return torch.cat(r, 0)
        elif target_type is None or return_type_dict[target_type] is None:
            return self.decoder_c2m(z_c2m), self.decoder_m2c(z_m2c), self.decoder_m2m(z_m2m)
        else:
            # For explanation, explaining a specific edge type
            return decode_type_dict[target_type](return_type_dict[target_type])

    def return_embedding_dict(self, x_dict, edge_index_dict):
        with torch.no_grad():
            emb = self.encoder(x_dict, edge_index_dict)
        return emb['city'], emb['mesh']


class MyGCN(HGTConv, ABC):

    def __init__(self,
                 in_channels: Union[int, Dict[str, int]],
                 out_channels: int,
                 metadata: Metadata,
                 heads: int,
                 **kwargs):
        super().__init__(in_channels, out_channels, metadata, heads, **kwargs)

    def _explain_message_with_masks(self,
                                    inputs: Tensor,
                                    dim_size: int,
                                    edge_mask: Tensor,
                                    loop_mask: Tensor) -> Tensor:
        if self._apply_sigmoid:
            edge_mask = edge_mask.sigmoid()

        if inputs.size(self.node_dim) != edge_mask.size(0):
            edge_mask = edge_mask[loop_mask]
            loop = edge_mask.new_ones(dim_size)
            edge_mask = torch.cat([edge_mask, loop], dim=0)
        assert inputs.size(self.node_dim) == edge_mask.size(0)

        size = [1] * inputs.dim()
        size[self.node_dim] = -1
        return inputs * edge_mask.view(size)

    def explain_message(self, inputs: Tensor, size_i: int) -> Tensor:
        edge_mask_dict = self._edge_mask
        loop_mask_dict = self._loop_mask
        edge_keys = self._edge_keys

        assert isinstance(edge_mask_dict, dict)

        edge_mask = torch.cat(
            [edge_mask_dict[edge_type] for edge_type in edge_keys])
        loop_mask = torch.cat(
            [loop_mask_dict[edge_type] for edge_type in edge_keys])

        return self._explain_message_with_masks(inputs, size_i, edge_mask, loop_mask)
