'''
using to test the difference between
'''
import argparse
import csv
import datetime
import glob
import importlib
import multiprocessing
import os
import pdb
import sys
import time
import warnings
from functools import partial

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from ldm.data.base import Txt2ImgIterableBaseDataset
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from packaging import version
from PIL import Image
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import (Callback, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.utilities import rank_zero_info
from torch import autocast
from torch.utils.data import DataLoader, Dataset, Subset, random_split

if version.parse(pl.__version__) > version.parse('1.4.2'):
    from pytorch_lightning.utilities import rank_zero_only
    from pytorch_lightning.plugins.precision import MixedPrecisionPlugin
else:
    from pytorch_lightning.utilities.distributed import rank_zero_only

warnings.filterwarnings('ignore')


def get_parser(**parser_kwargs):

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        '-n',
        '--name',
        type=str,
        const=True,
        default='',
        nargs='?',
        help='postfix for logdir',
    )
    parser.add_argument(
        '-r',
        '--resume',
        type=str,
        default='',
        nargs='?',
        help='resume from logdir or checkpoint in logdir',
    )
    parser.add_argument(
        '-b',
        '--base',
        nargs='*',
        metavar='base_config.yaml',
        help='paths to base configs. Loaded from left-to-right. '
        'Parameters can be overwritten or added with command-line options of the form `--key value`.',
        default=list(),
    )
    parser.add_argument(
        '-t',
        '--train',
        type=str2bool,
        const=True,
        default=False,
        nargs='?',
        help='train',
    )
    parser.add_argument(
        '--no-test',
        type=str2bool,
        const=True,
        default=False,
        nargs='?',
        help='disable test',
    )
    parser.add_argument(
        '-p', '--project', help='name of new or path to existing project')
    parser.add_argument(
        '-d',
        '--debug',
        type=str2bool,
        nargs='?',
        const=True,
        default=False,
        help='enable post-mortem debugging',
    )
    parser.add_argument(
        '-s',
        '--seed',
        type=int,
        default=23,
        help='seed for seed_everything',
    )
    parser.add_argument(
        '-f',
        '--postfix',
        type=str,
        default='',
        help='post-postfix for default name',
    )
    parser.add_argument(
        '-l',
        '--logdir',
        type=str,
        default='logs',
        help='directory for logging dat shit',
    )
    parser.add_argument(
        '--scale_lr',
        type=str2bool,
        nargs='?',
        const=True,
        default=True,
        help='scale base-lr by ngpu * batch_size * n_accumulate',
    )

    parser.add_argument(
        '--cfg',
        default=3.,
        type=float,
    )
    return parser


def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()

    dataset = worker_info.dataset
    worker_id = worker_info.id

    if isinstance(dataset, Txt2ImgIterableBaseDataset):
        split_size = dataset.num_records // worker_info.num_workers
        # reset num_records to the true number to retain reliable length information
        dataset.sample_ids = dataset.valid_ids[worker_id
                                               * split_size:(worker_id + 1)
                                               * split_size]
        current_id = np.random.choice(len(np.random.get_state()[1]), 1)
        return np.random.seed(np.random.get_state()[1][current_id] + worker_id)
    else:
        return np.random.seed(np.random.get_state()[1][0] + worker_id)


class DataModuleFromConfig(pl.LightningDataModule):

    def __init__(self,
                 batch_size,
                 train=None,
                 validation=None,
                 test=None,
                 predict=None,
                 wrap=False,
                 num_workers=None,
                 shuffle_test_loader=False,
                 use_worker_init_fn=False,
                 shuffle_val_dataloader=False):

        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else min(
            batch_size * 2, multiprocessing.cpu_count())

        self.use_worker_init_fn = use_worker_init_fn
        if train is not None:
            self.dataset_configs['train'] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs['validation'] = validation
            self.val_dataloader = partial(
                self._val_dataloader, shuffle=shuffle_val_dataloader)
        if test is not None:
            self.dataset_configs['test'] = test
            self.test_dataloader = partial(
                self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.dataset_configs['predict'] = predict
            self.predict_dataloader = self._predict_dataloader
        self.wrap = wrap

    def prepare_data(self):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)

    def setup(self, stage=None):
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])
        '''
        for i in range(100):
            self.datasets['train'][i+1000]
            pdb.set_trace()
        '''

    def _train_dataloader(self):
        is_iterable_dataset = isinstance(self.datasets['train'],
                                         Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(
            self.datasets['train'],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False if is_iterable_dataset else True,
            worker_init_fn=init_fn)

    def _val_dataloader(self, shuffle=False):

        if isinstance(self.datasets['validation'],
                      Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(
            self.datasets['validation'],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=init_fn,
            shuffle=shuffle)

    def _test_dataloader(self, shuffle=False):
        is_iterable_dataset = isinstance(self.datasets['test'],
                                         Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        # do not shuffle dataloader for iterable dataset
        shuffle = shuffle and (not is_iterable_dataset)

        return DataLoader(
            self.datasets['test'],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=init_fn,
            shuffle=shuffle)

    def _predict_dataloader(self, shuffle=False):
        if isinstance(self.datasets['predict'],
                      Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(
            self.datasets['predict'],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=init_fn)


class SetupCallback(Callback):

    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config,
                 lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    def on_keyboard_interrupt(self, trainer, pl_module):
        if trainer.global_rank == 0:
            print('Summoning checkpoint.')
            ckpt_path = os.path.join(self.ckptdir, 'last.ckpt')
            trainer.save_checkpoint(ckpt_path)

    def on_fit_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            if 'callbacks' in self.lightning_config:
                if 'metrics_over_trainsteps_checkpoint' in self.lightning_config[
                        'callbacks']:
                    os.makedirs(
                        os.path.join(self.ckptdir, 'trainstep_checkpoints'),
                        exist_ok=True)
            print('Project config')
            print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(
                self.config,
                os.path.join(self.cfgdir, '{}-project.yaml'.format(self.now)))

            print('Lightning config')
            print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(
                OmegaConf.create({'lightning': self.lightning_config}),
                os.path.join(self.cfgdir,
                             '{}-lightning.yaml'.format(self.now)))

        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, 'child_runs', name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass


class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(trainer.strategy.root_device.index)
        torch.cuda.synchronize(trainer.strategy.root_device.index)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        torch.cuda.synchronize(trainer.strategy.root_device.index)
        max_memory = torch.cuda.max_memory_allocated(
            trainer.strategy.root_device.index) / 2**20
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.training_type_plugin.reduce(max_memory)
            epoch_time = trainer.training_type_plugin.reduce(epoch_time)

            rank_zero_info(f'Average Epoch time: {epoch_time:.2f} seconds')
            rank_zero_info(f'Average Peak memory {max_memory:.2f}MiB')
        except AttributeError:
            pass


def load_pretrained_vae_weights(path):
    sd_weights = torch.load(path)
    vae_weights = dict()

    for key in sd_weights['state_dict'].keys():
        if 'first' in key:
            vae_weights[key.replace('first_stage_model.',
                                    '')] = sd_weights['state_dict'][key]

    torch.save({'state_dict': vae_weights}, 'sd_2_1_vae.ckpt')

    return vae_weights


if __name__ == '__main__':

    # weights = load_pretrained_vae_weights('./models/ldm/stable-diffusion-v1/sd-v1-4.ckpt')

    # custom parser to specify config files, train, test and debug mode,
    # postfix, resume.
    # `--key value` arguments are interpreted as arguments to the trainer.
    # `nested.key=value` arguments are interpreted as config parameters.
    # configs are merged from left-to-right followed by command line parameters.

    # model:
    #   base_learning_rate: float
    #   target: path to lightning module
    #   params:
    #       key: value
    # data:
    #   target: main.DataModuleFromConfig
    #   params:
    #      batch_size: int
    #      wrap: bool
    #      train:
    #          target: path to train dataset
    #          params:
    #              key: value
    #      validation:
    #          target: path to validation dataset
    #          params:
    #              key: value
    #      test:
    #          target: path to test dataset
    #          params:
    #              key: value
    # lightning: (optional, has sane defaults and can be specified on cmdline)
    #   trainer:
    #       additional arguments to trainer
    #   logger:
    #       logger to instantiate
    #   modelcheckpoint:
    #       modelcheckpoint to instantiate
    #   callbacks:
    #       callback1:
    #           target: importpath
    #           params:
    #               key: value

    now = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')

    # add cwd for convenience and to make classes in this file available when
    # running as `python main.py`
    # (in particular `main.DataModuleFromConfig`)
    sys.path.append(os.getcwd())

    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)

    opt, unknown = parser.parse_known_args()
    if opt.name and opt.resume:
        raise ValueError(
            '-n/--name and -r/--resume cannot be specified both.'
            'If you want to resume training in a new log folder, '
            'use -n/--name in combination with --resume_from_checkpoint')

    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError('Cannot find {}'.format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split('/')
            # idx = len(paths)-paths[::-1].index("logs")+1
            # logdir = "/".join(paths[:idx])
            logdir = '/'.join(paths[:-2])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip('/')
            ckpt = os.path.join(logdir, 'checkpoints', 'last.ckpt')

        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(
            glob.glob(os.path.join(logdir, 'configs/*.yaml')))
        opt.base = base_configs + opt.base
        _tmp = logdir.split('/')
        nowname = _tmp[-1]
    else:
        if opt.name:
            name = '_' + opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = '_' + cfg_name
        else:
            name = ''
        nowname = now + name + opt.postfix
        logdir = os.path.join(opt.logdir, nowname)

    ckptdir = os.path.join(logdir, 'checkpoints')
    cfgdir = os.path.join(logdir, 'configs')
    seed_everything(opt.seed)

    # ************************************************training**************************************************** #

    # init and save configs
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    lightning_config = config.pop('lightning', OmegaConf.create())
    # merge trainer cli with config
    trainer_config = lightning_config.get('trainer', OmegaConf.create())
    # default to ddp
    # trainer_config["accelerator"] = "ddp"
    if version.parse(pl.__version__) > version.parse('1.4.2'):
        trainer_config['accelerator'] = 'cuda'
        trainer_config['strategy'] = 'ddp'
    else:
        trainer_config['accelerator'] = 'ddp'

    lightning_config.callbacks.image_logger.params.sub_name = '_cfg_{:03d}'.format(
        int(opt.cfg))
    lightning_config.callbacks.image_logger.params.free_scale = float(opt.cfg)

    for k in nondefault_trainer_args(opt):
        trainer_config[k] = getattr(opt, k)
    if not 'gpus' in trainer_config:
        del trainer_config['accelerator']
        cpu = True
    else:
        gpuinfo = trainer_config['gpus']
        print(f'Running on GPUs {gpuinfo}')
        cpu = False

    trainer_opt = argparse.Namespace(**trainer_config)
    lightning_config.trainer = trainer_config

    model = instantiate_from_config(config.model)

    # trainer and callbacks
    trainer_kwargs = dict()

    # default logger configs
    if version.parse(pl.__version__) > version.parse('1.4.2'):
        default_logger_cfgs = {
            'wandb': {
                'target': 'pytorch_lightning.loggers.WandbLogger',
                'params': {
                    'name': nowname,
                    'save_dir': logdir,
                    'offline': True,
                    'id': nowname,
                }
            },
            'tensorboard': {
                'target': 'pytorch_lightning.loggers.TensorBoardLogger',
                'params': {
                    'name': 'tensorboard',
                    'save_dir': logdir,
                }
            },
        }
        default_logger_cfg = default_logger_cfgs['tensorboard']
    else:
        default_logger_cfgs = {
            'wandb': {
                'target': 'pytorch_lightning.loggers.WandbLogger',
                'params': {
                    'name': nowname,
                    'save_dir': logdir,
                    'offline': True,
                    'id': nowname,
                }
            },
            'testtube': {
                'target': 'pytorch_lightning.loggers.TestTubeLogger',
                'params': {
                    'name': 'testtube',
                    'save_dir': logdir,
                }
            },
        }
        default_logger_cfg = default_logger_cfgs['testtube']
    if 'logger' in lightning_config:
        logger_cfg = lightning_config.logger
    else:
        logger_cfg = OmegaConf.create()
    logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
    trainer_kwargs['logger'] = instantiate_from_config(logger_cfg)

    # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
    # specify which metric is used to determine best models
    default_modelckpt_cfg = {
        'target': 'pytorch_lightning.callbacks.ModelCheckpoint',
        'params': {
            'dirpath': ckptdir,
            'filename': '{epoch:06}',
            'verbose': True,
            'save_last': True,
        }
    }
    if hasattr(model, 'monitor'):
        print(f'Monitoring {model.monitor} as checkpoint metric.')
        default_modelckpt_cfg['params']['monitor'] = model.monitor
        default_modelckpt_cfg['params']['save_top_k'] = 3

    if 'modelcheckpoint' in lightning_config:
        modelckpt_cfg = lightning_config.modelcheckpoint
    else:
        modelckpt_cfg = OmegaConf.create()
    modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
    print(f'Merged modelckpt-cfg: \n{modelckpt_cfg}')
    if version.parse(pl.__version__) < version.parse('1.4.0'):
        trainer_kwargs['checkpoint_callback'] = instantiate_from_config(
            modelckpt_cfg)

    # add callback which sets up log directory
    default_callbacks_cfg = {
        'setup_callback': {
            'target': 'vae_main.SetupCallback',
            'params': {
                'resume': opt.resume,
                'now': now,
                'logdir': logdir,
                'ckptdir': ckptdir,
                'cfgdir': cfgdir,
                'config': config,
                'lightning_config': lightning_config,
            }
        },
        'image_logger': {
            'target': 'ldm.logger.image_logger.ImageLogger',
            'params': {
                'batch_frequency': 750,
                'max_images': 4,
                'clamp': True
            }
        },
        'learning_rate_logger': {
            'target': 'vae_main.LearningRateMonitor',
            'params': {
                'logging_interval': 'step',
                # "log_momentum": True
            }
        },
        'cuda_callback': {
            'target': 'vae_main.CUDACallback'
        },
    }

    if version.parse(pl.__version__) >= version.parse('1.4.0'):
        default_callbacks_cfg.update({'checkpoint_callback': modelckpt_cfg})

    if 'callbacks' in lightning_config:
        callbacks_cfg = lightning_config.callbacks
    else:
        callbacks_cfg = OmegaConf.create()

    if 'metrics_over_trainsteps_checkpoint' in callbacks_cfg:
        print(
            'Caution: Saving checkpoints every n train steps without deleting. This might require some free space.'
        )
        default_metrics_over_trainsteps_ckpt_dict = {
            'metrics_over_trainsteps_checkpoint': {
                'target': 'pytorch_lightning.callbacks.ModelCheckpoint',
                'params': {
                    'dirpath': os.path.join(ckptdir, 'trainstep_checkpoints'),
                    'filename': '{epoch:06}-{step:09}',
                    'verbose': True,
                    'save_top_k': -1,
                    'every_n_train_steps': 10000,
                    'save_weights_only': True
                }
            }
        }
        default_callbacks_cfg.update(default_metrics_over_trainsteps_ckpt_dict)

    callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
    if 'ignore_keys_callback' in callbacks_cfg and hasattr(
            trainer_opt, 'resume_from_checkpoint'):
        callbacks_cfg.ignore_keys_callback.params[
            'ckpt_path'] = trainer_opt.resume_from_checkpoint
    elif 'ignore_keys_callback' in callbacks_cfg:
        del callbacks_cfg['ignore_keys_callback']

    trainer_kwargs['callbacks'] = [
        instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg
    ]

    trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
    trainer.logdir = logdir  ###

    # dataset
    data = instantiate_from_config(config.data)

    # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
    # calling these ourselves should not be necessary but it is.
    # lightning still takes care of proper multiprocessing though
    data.prepare_data()
    data.setup()

    print('#### Data #####')
    for k in data.datasets:
        print(
            f'{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}'
        )

    # configure learning rate
    bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
    if not cpu:
        ngpu = len(lightning_config.trainer.gpus.strip(',').split(','))
    else:
        ngpu = 1
    if 'accumulate_grad_batches' in lightning_config.trainer:
        accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
    else:
        accumulate_grad_batches = 1
    print(f'accumulate_grad_batches = {accumulate_grad_batches}')
    lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches

    if opt.scale_lr:
        model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
        print(
            'Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)'
            .format(model.learning_rate, accumulate_grad_batches, ngpu, bs,
                    base_lr))
    else:
        model.learning_rate = base_lr
        print('++++ NOT USING LR SCALING ++++')
        print(f'Setting learning rate to {model.learning_rate:.2e}')

    # allow checkpointing via USR1
    def melk(*args, **kwargs):
        # run all checkpoint hooks
        if opt.train:
            if trainer.global_rank == 0:
                print('Summoning checkpoint.')
                ckpt_path = os.path.join(ckptdir, 'last.ckpt')
                trainer.save_checkpoint(ckpt_path)

    def divein(*args, **kwargs):
        if trainer.global_rank == 0:
            import pudb
            pudb.set_trace()

    import signal
    signal.signal(signal.SIGUSR1, melk)
    signal.signal(signal.SIGUSR2, divein)

    # test loading ckpt
    if not opt.train:
        pl_sd = torch.load(opt.resume, map_location='cpu')
        if 'global_step' in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        sd = pl_sd['state_dict']
        m, u = model.load_state_dict(sd, strict=False)
        if len(m) > 0:
            print('missing keys:')
            print(m)
        if len(u) > 0:
            print('unexpected keys:')
            print(u)

        del pl_sd

    # run
    if opt.train:
        try:
            trainer.fit(model, data)
        except Exception:
            melk()
            raise
    if not opt.no_test and not trainer.interrupted:
        with autocast('cuda'):
            trainer.test(model, data)
