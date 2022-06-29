from pathlib import Path
from typing import List
import numpy as np
import hydra
import omegaconf
import pytorch_lightning as pl
from omegaconf import DictConfig  # , OmegaConf
from pytorch_lightning import seed_everything, Callback
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger

from graphwm.common import log_hyperparameters, PROJECT_ROOT

def compile_expname(cfg: DictConfig):
  expname = f'{cfg.data.name}_{cfg.model.name}'
  return expname

def build_callbacks(cfg: DictConfig) -> List[Callback]:
  callbacks: List[Callback] = []
    
  if "lr_monitor" in cfg.logging:
    hydra.utils.log.info("Adding callback <LearningRateMonitor>")
    callbacks.append(
        LearningRateMonitor(
            logging_interval=cfg.logging.lr_monitor.logging_interval,
            log_momentum=cfg.logging.lr_monitor.log_momentum,
        )
    )

  if "early_stopping" in cfg.train:
    hydra.utils.log.info("Adding callback <EarlyStopping>")
    callbacks.append(
        EarlyStopping(
            monitor=cfg.train.monitor_metric,
            mode=cfg.train.monitor_metric_mode,
            patience=cfg.train.early_stopping.patience,
            verbose=cfg.train.early_stopping.verbose,
        )
    )

  if "model_checkpoints" in cfg.train:
    hydra.utils.log.info("Adding callback <ModelCheckpoint>")
    callbacks.append(
        ModelCheckpoint(
            dirpath=cfg.workdir,
            monitor=cfg.train.monitor_metric,
            mode=cfg.train.monitor_metric_mode,
            save_top_k=cfg.train.model_checkpoints.save_top_k,
            save_last=cfg.train.model_checkpoints.save_last,
            verbose=cfg.train.model_checkpoints.verbose,
        )
    )

  return callbacks


def run(cfg: DictConfig) -> None:
    """
    Generic train loop

    :param cfg: run configuration, defined by Hydra in /conf
    """
    if cfg.train.deterministic:
      seed_everything(cfg.train.random_seed)

    if cfg.train.pl_trainer.fast_dev_run:
      hydra.utils.log.info(
          f"Debug mode <{cfg.train.pl_trainer.fast_dev_run=}>. "
          f"Forcing debugger friendly configuration!"
      )
      # Debuggers don't like GPUs nor multiprocessing
      cfg.train.pl_trainer.gpus = 0
      cfg.data.datamodule.num_workers.train = 0
      cfg.data.datamodule.num_workers.val = 0
      cfg.data.datamodule.num_workers.test = 0
      # Switch wandb mode to offline to prevent online logging
      cfg.logging.wandb.mode = "offline"

    # Hydra run directory
    hydra_dir = Path(cfg.workdir)

    # Instantiate datamodule
    hydra.utils.log.info(f"Instantiating <{cfg.data.datamodule._target_}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.data.datamodule, _recursive_=False
    )

    # Instantiate model
    hydra.utils.log.info(f"Instantiating <{cfg.model._target_}>")
    model: pl.LightningModule = hydra.utils.instantiate(
        cfg.model,
        optim=cfg.optim,
        data=cfg.data,
        logging=cfg.logging,
        _recursive_=False,
    )

    # Instantiate the callbacks
    callbacks: List[Callback] = build_callbacks(cfg=cfg)

    # Logger instantiation/configuration
    wandb_logger = None
    if "wandb" in cfg.logging:
        hydra.utils.log.info("Instantiating <WandbLogger>")
        wandb_config = cfg.logging.wandb
        wandb_logger = WandbLogger(
            **wandb_config,
            tags=cfg.core.tags,
        )
        hydra.utils.log.info(f"W&B is now watching <{cfg.logging.wandb_watch.log}>!")
        wandb_logger.watch(
            model,
            log=cfg.logging.wandb_watch.log,
            log_freq=cfg.logging.wandb_watch.log_freq,
        )

    # Load checkpoint (if exist)
    workdir = Path(cfg.workdir)
    if (workdir / 'last.ckpt').exists():
      ckpt = str(workdir / 'last.ckpt')
    else:
      ckpts = list(workdir.glob('*.ckpt'))
      if len(ckpts) > 0:
        ckpt_epochs = np.array([int(ckpt.parts[-1].split('-')[0].split('=')[1]) for ckpt in ckpts])
        ckpt = str(ckpts[ckpt_epochs.argsort()[-1]])
        hydra.utils.log.info(f"found checkpoint: {ckpt}")
      else:
        ckpt = None
            
    hydra.utils.log.info("Instantiating the Trainer")

    # The Lightning core, the Trainer
    trainer = pl.Trainer(
        default_root_dir=hydra_dir,
        logger=wandb_logger,
        callbacks=callbacks,
        deterministic=cfg.train.deterministic,
        val_check_interval=cfg.logging.val_check_interval,
        plugins=DDPPlugin(find_unused_parameters=False),
        progress_bar_refresh_rate=cfg.logging.progress_bar_refresh_rate,
        resume_from_checkpoint=ckpt,
        **cfg.train.pl_trainer,
    )
    
    log_hyperparameters(trainer=trainer, model=model, cfg=cfg)

    hydra.utils.log.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule)

    hydra.utils.log.info("Starting testing!")
    trainer.test(datamodule=datamodule)

    # Logger closing to release resources/avoid multi-run conflicts
    if wandb_logger is not None:
        wandb_logger.experiment.finish()


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="train")
def main(cfg: omegaconf.DictConfig):
    run(cfg)

if __name__ == "__main__":
    main()