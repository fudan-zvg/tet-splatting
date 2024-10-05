import os
import pdb

from pytorch_lightning.callbacks import (Callback, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.utilities.distributed import rank_zero_only


class GlobalCallback(Callback):

    def __init__(self,
                 resume,
                 now,
                 logdir,
                 ckptdir,
                 cfgdir,
                 config,
                 accumulate_grad_batches,
                 lightning_config,
                 save_steps=20000,
                 save_start=False):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config
        self.accumulate_grad_batches = accumulate_grad_batches
        self.save_steps = save_steps
        self.save_global = self.save_steps // self.accumulate_grad_batches
        self.save_start = save_start

    @rank_zero_only
    def on_train_batch_end(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        outputs,
        batch,
        batch_idx,
    ) -> None:
        global_step = pl_module.global_step
        if global_step > 0 and global_step % self.save_global == 0:
            ckpt_path = os.path.join(self.ckptdir,
                                     'step_{:08d}.ckpt'.format(global_step))
            trainer.save_checkpoint(ckpt_path)

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        if self.save_start:
            global_step = pl_module.global_step
            ckpt_path = os.path.join(self.ckptdir,
                                     'step_{:08d}.ckpt'.format(global_step))
            trainer.save_checkpoint(ckpt_path)
