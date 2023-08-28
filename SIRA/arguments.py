import argparse


class Arguments:

    def __init__(self):
        self._parser = argparse.ArgumentParser(description='Arguments.')

        # Dataset arguments
        self._parser.add_argument('--dataset_path',
                                  type=str,
                                  default='dataset/synth2real',
                                  help='Dataset file path.')
        self._parser.add_argument('--real_data_root',
                                  type=str,
                                  default='../PCR/dataset/3DMatch/data/train',
                                  help='Dataset file path.')
        self._parser.add_argument('--synth_data_root',
                                  type=str,
                                  default='../PCR/dataset/FlyingShapes',
                                  help='Dataset file path.')
        self._parser.add_argument('--batch_size',
                                  type=int,
                                  default=1,
                                  help='Integer value for batch size.')
        self._parser.add_argument('--num_workers',
                                  type=int,
                                  default=20,
                                  help='Integer value for batch size.')
        self._parser.add_argument(
            '--n_neighbors',
            type=int,
            default=30,
            help='Integer value for number of neighbors.')
        self._parser.add_argument(
            '--voxelsize',
            type=float,
            default=0.025,
            help='value for search voxelsize of pointcloud.')
        self._parser.add_argument(
            '--radius',
            type=float,
            default=0.0625,
            help=
            'value for search radius of neighbors.(default: 2.5 * voxelsize)')
        self._parser.add_argument(
            '--sigma',
            type=float,
            default=0.05,
            help=
            'value for influence radius of kernel.(default: 2.0 * voxelsize)')

        # Training arguments
        self._parser.add_argument('--gpu',
                                  type=int,
                                  default=0,
                                  help='GPU number to use.')
        self._parser.add_argument('--epochs',
                                  type=int,
                                  default=200,
                                  help='Integer value for epochs.')
        self._parser.add_argument('--lr',
                                  type=float,
                                  default=1e-4,
                                  help='Float value for learning rate.')
        self._parser.add_argument('--ckpt_path',
                                  type=str,
                                  default='experiment/synth2real/ckpt/',
                                  help='Checkpoint path.')
        self._parser.add_argument('--ckpt_save',
                                  type=str,
                                  default='ckpt_',
                                  help='Checkpoint name to save.')
        self._parser.add_argument(
            '--ckpt_load',
            type=str,
            help='Checkpoint name to load. (default:None)')
        self._parser.add_argument('--result_path',
                                  type=str,
                                  default='experiment/synth2real/generated/',
                                  help='Generated results path.')
        self._parser.add_argument('--result_save',
                                  type=str,
                                  default='pc_epoch',
                                  help='Generated results name to save.')
        self._parser.add_argument('--summary_path',
                                  type=str,
                                  default='experiment/synth2real/summary',
                                  help='Summary path')

        # Network arguments
        self._parser.add_argument('--En_iter',
                                  type=int,
                                  default=1,
                                  help='Number of iterations for generator.')
        self._parser.add_argument(
            '--D_iter',
            type=int,
            default=1,
            help='Number of iterations for discriminator.')
        self._parser.add_argument('--G_iter',
                                  type=int,
                                  default=1,
                                  help='Number of iterations for generator.')
        self._parser.add_argument('--D_FEAT',
                                  type=int,
                                  default=[
                                      256,
                                      512,
                                      1024,
                                      2048,
                                  ],
                                  nargs='+',
                                  help='Features for discriminator.')

    def parser(self):
        return self._parser
