import argparse


def add_bool_arg(p, name, default=True):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=name, action='store_true')
    group.add_argument('--no_' + name, dest=name, action='store_false')
    p.set_defaults(**{name: default})


parser = argparse.ArgumentParser()

# ---------------------------File--------------------------- #
parser.add_argument('--data_path',           default='./data')
parser.add_argument('--explain_path',        default='./explain')


# ---------------------------Train--------------------------- #
parser.add_argument('--experiment_name',                        default=None)
parser.add_argument('--device',                                 default=None)
parser.add_argument('--seed',                      type=int,    default=42)
parser.add_argument('--hidden_channels',           type=int,    default=256)
parser.add_argument('--embedding_size',            type=int,    default=128)
parser.add_argument('--lr',                        type=float,  default=0.0005)
parser.add_argument('--lr_weight_decay',           type=float,  default=0.001)
parser.add_argument('--epochs',                    type=int,    default=40000)
parser.add_argument('--dropout',                   type=float,  default=0.3)
parser.add_argument('--heads',                     type=int,    default=8)
parser.add_argument('--layer_type',                             default='HGT')
parser.add_argument('--num_layer',                 type=int,    default=3)
parser.add_argument('--scheduler_step',            type=int,    default=2000)
parser.add_argument('--scheduler_gamma',           type=float,  default=0.95)
parser.add_argument('--early_stopper_patience',    type=int,    default=1000)
parser.add_argument('--early_stopper_delta',       type=float,  default=500.)
parser.add_argument('--clip_threshold',            type=float,  default=None)
parser.add_argument('--evaluate_epoch_interval',   type=int,    default=50)
add_bool_arg(parser, 'scheduled', default=True)


# ---------------------------Feature------------------------- #
# parser.add_argument('--city_feature',                           default='onehot')
# parser.add_argument('--mesh_feature',                           default='real')


# ---------------------------Message Passing Type------------------------- #
add_bool_arg(parser, 'inclusion')
add_bool_arg(parser, 'geo', default=False)
add_bool_arg(parser, 'flow')


# ---------------------------Model--------------------------- #
parser.add_argument('--loss_type',                             default='weighted_focal_mse')
parser.add_argument('--multi_task_weight_m2m',   type=float,    default=0.8)


# ---------------------------Explain------------------------- #

parser.add_argument('--explain_threshold',       type=int,      default=200)

# ---------------------------Save--------------------------- #
parser.add_argument('--save_folder',                           default='./result')


args = parser.parse_args()
