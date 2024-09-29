import torch
import logging
import pandas as pd
from torch_geometric.data import HeteroData
from preprocessing import preprocessing


class ODGraph:
    def __init__(self):
        self.digits = {'city': 5, 'mesh': 9, 'pref': 2, 'region': 3}
        self.name = None
        # For node IDs and mapped indices
        self.nodes = {}
        self.xs = {}
        self.edges = {}
        self.graph = None
        self.logger = logging.getLogger(self.__class__.__name__)

    def log_graph_info(self):
        """
        Log information about the generated graph.
        """
        if self.graph is None:
            self.logger.error("Graph has not been generated yet. Call generate_graph() first.")
            return
        self.logger.info('============== Complete Graph Info ===============')
        self.logger.info(f'#Nodes: {self.graph.num_nodes}')
        self.logger.info(f'#Edges: {self.graph.num_edges}')
        self.logger.info(f'#Node features: {self.graph.num_node_features}')
        self.logger.info(f'Has self loops: {self.graph.has_self_loops()}')
        # self.logger.info(f'Shape of edge index: {self.graph.edge_index.shape}')

    def add_node_list(self, nodes):
        """

        Args:
            nodes: List of nodes -> pd.Series

        Returns:

        """
        unique_id = nodes.unique()
        assert len(set(pd.Series(unique_id).str.len())) == 1
        inv_map = {v: k for k, v in self.digits.items()}
        name = inv_map[len(unique_id[0])]
        self.nodes[name] = pd.DataFrame(data={'rawId': unique_id,
                                              'mappedId': pd.RangeIndex(len(unique_id))})
        self.nodes[name] = dict(zip(self.nodes[name]['rawId'], self.nodes[name]['mappedId']))

    def mapping_nodes_to_raw_id(self, mapped_id, name='mesh'):
        a = self.nodes[name]
        inv_map = {v: k for k, v in a.items()}
        return mapped_id.map(inv_map)


class HeterogeneousODGraph(ODGraph):
    def __init__(self, xs, od, inclusion, neighbor=None):
        super().__init__()
        self.edges['od'] = od
        self.edges['inclusion'] = inclusion
        if neighbor is not None:
            self.edges['neighbor'] = neighbor
        self.xs['mesh'] = xs['mesh']
        self.xs['city'] = xs['city']
        self.graph = HeteroData()
        self.generate_graph()
        super().log_graph_info()

    def generate_graph(self):
        # Mapping nodes to the unique ids
        self.add_node_list(self.edges['inclusion']['KEY_CODE'])
        self.add_node_list(self.edges['inclusion']['city_code'])

        self.graph['mesh'].x = torch.tensor(self.xs['mesh'].values, dtype=torch.float)
        self.graph['city'].x = torch.eye(self.xs['city'].shape[1], dtype=torch.float)
        for name in self.nodes.keys():
            if name in ['mesh', 'city']:
                continue
            self.graph[name].x = torch.randn(len(self.nodes[name]), self.xs['mesh'].values.shape[1], dtype=torch.float)

        # Edges
        mask = (self.edges['od']['origin_key'].str.len() == self.digits['mesh']) & \
               (self.edges['od']['dest_key'].str.len() == self.digits['mesh'])
        od = self.edges['od'].loc[mask]
        d = self.nodes
        edge_index = torch.stack([torch.tensor(od['origin_key'].map(d['mesh']).values),
                                 torch.tensor(od['dest_key'].map(d['mesh']).values)],
                                 dim=0)
        self.graph['mesh', 'm2m', 'mesh'].edge_index = edge_index
        self.graph['mesh', 'm2m', 'mesh'].edge_label = torch.tensor(od['num'].values)
        mask = (self.edges['od']['origin_key'].str.len() == self.digits['mesh']) & \
               (self.edges['od']['dest_key'].str.len() == self.digits['city'])
        od = self.edges['od'].loc[mask]
        edge_index = torch.stack([torch.tensor(od['origin_key'].map(d['mesh']).values),
                                 torch.tensor(od['dest_key'].map(d['city']).values)],
                                 dim=0)
        self.graph['mesh', 'm2c', 'city'].edge_index = edge_index
        self.graph['mesh', 'm2c', 'city'].edge_label = torch.tensor(od['num'].values)
        mask = (self.edges['od']['origin_key'].str.len() == self.digits['city']) &\
               (self.edges['od']['dest_key'].str.len() == self.digits['mesh'])
        od = self.edges['od'].loc[mask]
        edge_index = torch.stack([torch.tensor(od['origin_key'].map(d['city']).values),
                                 torch.tensor(od['dest_key'].map(d['mesh']).values)],
                                 dim=0)
        self.graph['city', 'c2m', 'mesh'].edge_index = edge_index
        self.graph['city', 'c2m', 'mesh'].edge_label = torch.tensor(od['num'].values)

        self.graph['city', 'include', 'mesh'].edge_index = torch.stack(
            [torch.tensor(self.edges['inclusion']['city_code'].map(d['city']).values),
             torch.tensor(self.edges['inclusion']['KEY_CODE'].map(d['mesh']).values)], dim=0)

        self.graph['mesh', 'isin', 'city'].edge_index = torch.stack(
            [torch.tensor(self.edges['inclusion']['KEY_CODE'].map(d['mesh']).values),
             torch.tensor(self.edges['inclusion']['city_code'].map(d['city']).values)], dim=0)

        if 'neighbor' in self.edges.keys():
            self.graph['mesh', 'near', 'mesh'].edge_index = torch.stack(
                [torch.tensor(self.edges['neighbor']['m1'].map(d['mesh']).values),
                 torch.tensor(self.edges['neighbor']['m2'].map(d['mesh']).values)], dim=0)

        assert self.graph['mesh'].num_features == 43 or 86
        assert self.graph['city'].num_features == 43 or 86
        assert self.graph['city'].num_nodes == 43

    def convert_edge_index_with_edge_value_to_df_in_one_edge_type(self, edge_index, type_pair, edge_value=None):
        assert edge_index.shape[0] == 2
        assert isinstance(type_pair, list)
        assert len(type_pair) == 2
        assert type_pair[0] in self.digits.keys()
        assert type_pair[1] in self.digits.keys()

        df = pd.concat([
            self.mapping_nodes_to_raw_id(pd.Series(edge_index[0].cpu().numpy()), name=type_pair[0]),
            self.mapping_nodes_to_raw_id(pd.Series(edge_index[1].cpu().numpy()), name=type_pair[1])
        ], axis=1)
        if edge_value is not None:
            if not isinstance(edge_value, list):
                edge_value = [edge_value]
        for v in edge_value:
            assert edge_index.shape[1] == v.shape[0]
            df = pd.concat([df, pd.Series(v.cpu().numpy())], axis=1)
        return df

    def convert_split_edge_index_with_edge_value(self, data, edge_values=None):
        e = {'c2m': [], 'm2c': [], 'm2m': []}
        if edge_values is None:
            e['c2m'].append(data['c2m'].edge_label)
            e['m2c'].append(data['m2c'].edge_label)
            e['m2m'].append(data['m2m'].edge_label)
        else:
            if not isinstance(edge_values, list):
                edge_values = [edge_values]
            for v in edge_values:
                e['c2m'].append(v['c2m'])
                e['m2c'].append(v['m2c'])
                e['m2m'].append(v['m2m'])

        df = pd.concat(
            [self.convert_edge_index_with_edge_value_to_df_in_one_edge_type(
                data['mesh', 'm2c', 'city'].edge_label_index, ['mesh', 'city'], e['m2c']),
             self.convert_edge_index_with_edge_value_to_df_in_one_edge_type(
                data['city', 'c2m', 'mesh'].edge_label_index, ['city', 'mesh'], e['c2m']),
             self.convert_edge_index_with_edge_value_to_df_in_one_edge_type(
                data['mesh', 'm2m', 'mesh'].edge_label_index, ['mesh', 'mesh'], e['m2m'])], axis=0)
        return df

    def filter_edge_index_with_target_node(self, target_node):
        assert isinstance(target_node, str)
        filter_result = {}
        if len(target_node) == self.digits['city']:
            edge_flow = self.edges['od']['c2m']
            filter_result['c2m'] = edge_flow[edge_flow['dest'] == target_node]
            edge_flow = self.edges['od']['m2c']
            filter_result['m2c'] = edge_flow[edge_flow['dest'] == target_node]
        elif len(target_node) == self.digits['mesh']:
            edge_flow = self.edges['od']['c2m']
            filter_result['c2m'] = edge_flow[edge_flow['dest'] == target_node]
            edge_flow = self.edges['od']['c2m']
            filter_result['m2c'] = edge_flow[edge_flow['dest'] == target_node]
            edge_flow = self.edges['od']['c2m']
            filter_result['m2m'] = edge_flow[edge_flow['dest'] == target_node]
        else:
            raise NotImplementedError

        return filter_result


if __name__ == '__main__':
    dict_df_data = preprocessing()
    g = HeterogeneousODGraph(dict_df_data['xs'], dict_df_data['edges']['od'], dict_df_data['edges']['include'])
    print(g.graph)
