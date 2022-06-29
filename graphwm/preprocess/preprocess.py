"""
clean up and rename to polymer.py in the end.
"""
import os
import sys
import time
from pathlib import Path
import multiprocessing as mp
from p_tqdm import p_umap

from battery import load_battery_data
from chain import load_polymer_rg

from graphwm.data.utils import store_data

def polymer_to_h5(data_dir, data_save_dir):
  """
  save whole trajectory data directly from .txt polymer trajectory.
  """
  data_dir = Path(data_dir)
  poly_file_dirs = [d for d in list(data_dir.iterdir()) if os.path.isdir(d)]
  print(f"Found {len(poly_file_dirs)} polymer trajectories.")
  print(f"Use {mp.cpu_count()} cores.")
  print("Start processing...")

  def process_one_file(poly_file):
    poly_index = poly_file.parts[-1]
    os.makedirs(os.path.join(data_save_dir, poly_index), exist_ok=True)
    if not Path(str(os.path.join(data_save_dir, poly_index, 'bond.h5'))).exists():
      try:
        data = load_polymer_rg(poly_file)
        store_data(['position'], [data[0]], os.path.join(data_save_dir, poly_index, 'position.h5'))
        store_data(['lattice'], [data[1]], os.path.join(data_save_dir, poly_index, 'lattice.h5'))
        store_data(['rgs'], [data[2]], os.path.join(data_save_dir, poly_index, 'rgs.h5'))
        store_data(['particle_type'], [data[3]], os.path.join(data_save_dir, poly_index, 'ptype.h5'))
        store_data(['bond_indices'], [data[4]], os.path.join(data_save_dir, poly_index, 'bond.h5'))
      except Exception as e:
        print(poly_index)
        print(e)
        pass

  now = time.time()
  process_one_file(poly_file_dirs[0])
  p_umap(process_one_file, poly_file_dirs)  
  elapsed = time.time() - now
  print(f"Done. Number of rollouts: {len(poly_file_dirs)} || Time Elapsed: {elapsed}")

def battery_to_h5(data_dir, data_save_dir):
  data_dir = Path(data_dir)
  poly_file_dirs = [d for d in list(data_dir.iterdir()) if os.path.isdir(d)]
  print(f"Found {len(poly_file_dirs)} polymer trajectories.")
  print(f"Use {mp.cpu_count()} cores.")
  print("Start processing...")
  
  def process_one_file(poly_file):
    poly_index = poly_file.parts[-1]
    os.makedirs(os.path.join(data_save_dir, poly_index), exist_ok=True)
    if not Path(str(os.path.join(data_save_dir, poly_index, 'diffusivity.h5'))).exists():
      try:
        data = load_battery_data(poly_file)
        store_data(['wrapped_position'], [data[0]], os.path.join(data_save_dir, poly_index, 'wrapped_position.h5'))
        store_data(['unwrapped_position'], [data[1]], os.path.join(data_save_dir, poly_index, 'unwrapped_position.h5'))
        store_data(['lattice'], [data[2]], os.path.join(data_save_dir, poly_index, 'lattice.h5'))
        store_data(['raw_particle_type'], [data[3]], os.path.join(data_save_dir, poly_index, 'raw_ptype.h5'))
        store_data(['particle_type'], [data[4]], os.path.join(data_save_dir, poly_index, 'ptype.h5'))
        store_data(['bond_indices'], [data[5]], os.path.join(data_save_dir, poly_index, 'bond.h5'))
        store_data(['bond_type'], [data[6]], os.path.join(data_save_dir, poly_index, 'bond_type.h5'))
        store_data(['diffusivity'], [data[7]], os.path.join(data_save_dir, poly_index, 'diffusivity.h5'))
      except OSError:
        pass
    
  now = time.time()
  p_umap(process_one_file, poly_file_dirs)  
  elapsed = time.time() - now
  print(f"Done. Number of rollouts: {len(poly_file_dirs)} || Time Elapsed: {elapsed}")

def protein_to_h5(data_dir, data_save_dir):
  pass

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('[Usage] arg1: dataset (chain/battery) arg2: data_dir arg3: save_dir')
        sys.exit(1)
    dataset, data_dir, data_save_dir = sys.argv[1:]
    if dataset == 'chain':
      polymer_to_h5(data_dir, data_save_dir)
    elif dataset == 'battery':
      battery_to_h5(data_dir, data_save_dir)
      