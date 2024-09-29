import os
import pandas as pd
import jismesh.utils as ju

from args import args
from feature import get_feature


def get_od(path):
    return pd.read_csv(path, header=0, encoding='shift-jis', dtype={'origin_key': str, 'dest_key': str, 'num': float})


def get_include(path):
    # read city-grid-inclusion file
    city_mesh_relation = pd.read_csv(path,
                                     encoding='shift-jis',
                                     names=['city_code', 'city_name', 'mesh_code'],
                                     header=0).astype({'city_code': int, 'mesh_code': int})
    # The data missing one mesh
    city_mesh_relation = pd.concat([city_mesh_relation,
                                    pd.DataFrame([{'city_code': 22103,
                                                   'city_name': '静岡市清水区',
                                                   'mesh_code': 52384420}])],
                                   ignore_index=True)

    # expand mesh level to 500m (default: 1000m)
    city_mesh_relation['KEY_CODE'] = city_mesh_relation['mesh_code'].apply(lambda x: list(ju.to_intersects(x, 4)))
    city_mesh_relation = city_mesh_relation.explode('KEY_CODE').reset_index().drop(['index',
                                                                                    'city_name',
                                                                                    'mesh_code'], axis=1)
    return city_mesh_relation.astype({'city_code': str, 'KEY_CODE': str})


def get_neighbor(path):
    neighbor = pd.read_csv(path, names=['m1', 'm2'])
    return neighbor.astype({'m1': str, 'm2': str})


def get_node_feature(df_edges, feature):
    if 'KEY_CODE' in df_edges.columns:
        # df_edges is inclusion edges
        city_feature = df_edges.merge(feature, on='KEY_CODE', how='left')
        mesh_feature = city_feature.iloc[:, 1:].copy().set_index('KEY_CODE')
        assert 'city_code' not in mesh_feature.columns
        assert city_feature.loc[city_feature['population'].isna()].empty
        city_feature = city_feature.drop(['KEY_CODE'], axis=1).groupby(['city_code']).sum()

        assert len(mesh_feature.columns) == 43
        assert len(city_feature.columns) == 43
        return city_feature, mesh_feature
    else:
        # df_edges is od edges
        df_unique_nodes = pd.Series(pd.concat([df_edges['origin_key'],
                                               df_edges['dest_key']]).unique(), name='KEY_CODE').to_frame()
        return df_unique_nodes.merge(feature, on='KEY_CODE', how='left')


def preprocessing():
    dict_path = {'include': os.path.join(args.data_path, 'meshCode', 'mesh_code_shizuoka.csv'),
                 'od': os.path.join(args.data_path, 'od', 'od.csv'),
                 'neighbor': os.path.join(args.data_path, 'geoNeighbor', 'edge_list_geo_neighbor.csv')}
    all_feature = get_feature().astype({'KEY_CODE': str})

    edges = {'od': get_od(dict_path['od']),
             'include': get_include(dict_path['include']),
             'neighbor': get_neighbor(dict_path['neighbor'])}
    xs = {'city': (get_node_feature(edges['include'], all_feature))[0],
          'mesh': (get_node_feature(edges['include'], all_feature))[1]}

    return {'edges': edges,
            'xs': xs}


if __name__ == '__main__':
    preprocessing()

    mesh = preprocessing()['xs']['mesh']
    city = preprocessing()['xs']['city']
    od = preprocessing()['edges']['od']
    inclusion = preprocessing()['edges']['include']
    geo_neighbor = preprocessing()['edges']['neighbor']
    mesh.to_csv(os.path.join(args.save_folder, 'result', 'input', 'mesh_feature_selected.csv'))
    city.to_csv(os.path.join(args.save_folder, 'result', 'input', 'city_feature_selected.csv'))
    od.to_csv(os.path.join(args.save_folder, 'result', 'input', 'od.csv'), index=False)
    inclusion.to_csv(os.path.join(args.save_folder, 'result', 'input', 'inclusion.csv'), index=False)
