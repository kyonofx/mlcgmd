import os
import time
import hydra
import numpy as np
import torch

from pathlib import Path
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything

from graphwm.data.datamodule import worker_init_fn
from graphwm.data.utils import dict_collate_fn
from graphwm.common import PROJECT_ROOT
from graphwm.model import GNS, PnR

MODELS = {
    'gns': GNS,
    'pnr': PnR
}

def run_eval(cfg):
  seed_everything(cfg.random_seed)
    
  model_dir = Path(cfg.model_dir)
  dataclass, modelclass = model_dir.parts[-1].split('_')[:2]
  save_dir = Path(cfg.save_dir)
  
  if modelclass == 'pnr':
    folder_name = f'nsteps{cfg.ld_kwargs.step_per_sigma}_stepsize_{cfg.ld_kwargs.step_size}'
  else:
    folder_name = 'rollouts'
  if (save_dir / folder_name / f'seed_{cfg.random_seed}.pt').exists():
    print('Rollout already exists.')
    return
  
  last_model_dir = Path(cfg.model_dir) / 'last.ckpt'
  if last_model_dir.exists():
    ckpt_path = str(last_model_dir)
  else:   
    ckpts = list(Path(cfg.model_dir).glob('*.ckpt'))
    ckpt_epochs = np.array([int(ckpt.parts[-1].split('-')[0].split('=')[1]) for ckpt in ckpts])
    ckpt_path = str(ckpts[ckpt_epochs.argsort()[-1]])
    
  print(f'load checkpoint: {ckpt_path}')

  model = MODELS[modelclass].load_from_checkpoint(ckpt_path)
  # prepare data
  dataset = hydra.utils.instantiate(cfg.data, 
                                    seq_len=model.hparams.seq_len, 
                                    dilation=model.hparams.dilation, 
                                    grouping=model.hparams.cg_level)
  data_loader = DataLoader(dataset, shuffle=False, batch_size=cfg.batch_size, num_workers=8, 
                          worker_init_fn=worker_init_fn, collate_fn=dict_collate_fn)
  
  model = model.to('cuda')
  model.eval()
  os.makedirs(save_dir, exist_ok=True)
  outputs = []
  
  # adjust ld_kwargs
  if modelclass == 'pnr':
    model.hparams.step_per_sigma = cfg.ld_kwargs.step_per_sigma
    model.hparams.step_size = cfg.ld_kwargs.step_size
  
  now = time.time()
  
  last_component = 0
  last_cluster = 0
  with torch.no_grad():
    for idx, batch in enumerate(data_loader):
      if idx == cfg.num_batches:
        break
      batch = {k: v.cuda() for k, v in batch.items()}
      simulate_steps = cfg.rollout_length // model.hparams.dilation
      output = model.simulate(batch, simulate_steps - model.hparams.seq_len, 
                              save_positions=cfg.save_pos, deter=cfg.deter, 
                              save_frequency=cfg.save_frequency)
      output.update({k: v.detach().cpu() for k, v in batch.items()})
      
      if model.hparams.cg_level > 1:
        output['cluster'] += last_cluster  # fix the offests for <cluster>
        last_cluster = output['cluster'].max()+1 
        if 'component' in output:
          output['component'] += last_component  # fix the offsets for <component>
          last_component = output['component'].max()+1    
      outputs.append(output)
  elapsed = time.time() - now
  
  # for output in range(cfg.num_batches):
  outputs = {k: torch.cat([d[k] for d in outputs]) for k in outputs[-1].keys()}
  outputs['time_elapsed'] = elapsed
  outputs['model_params'] = model.hparams
  outputs['eval_cfg'] = cfg
  
  os.makedirs(save_dir / folder_name, exist_ok=True)
  torch.save(outputs, save_dir / folder_name / f'seed_{cfg.random_seed}.pt')
  
  print(f'Finished {cfg.batch_size*cfg.num_batches} rollouts of {cfg.rollout_length} steps.')
  
@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="eval")
def main(cfg):
  run_eval(cfg)

if __name__ == "__main__":
  main()
