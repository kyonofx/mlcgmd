import warnings
import torch

from torch.optim.lr_scheduler import _LRScheduler

class CustomScheduleLR(_LRScheduler):
  """
  the learning rate schedule used in GNS.
  """
  def __init__(self, optimizer, min_lr, decay_steps, decay_rate, last_epoch=-1, verbose=False):
    self._min_lr = min_lr
    self._decay_steps = decay_steps
    self._decay_rate = decay_rate
    super(CustomScheduleLR, self).__init__(optimizer, last_epoch, verbose)
    
  def get_lr(self):
    if not self._get_lr_called_within_step:
        warnings.warn("To get the last learning rate computed by the scheduler, "
                      "please use `get_last_lr()`.", UserWarning)
    return [(self._min_lr + (self.base_lrs[0] - self._min_lr) * 
            self._decay_rate ** (self._step_count / self._decay_steps))]

  def _get_closed_form_lr(self):
    return [(self._min_lr + (self.base_lrs[0] - self._min_lr) * 
            self._decay_rate ** (self._step_count / self._decay_steps))]
      
class StandardScalerTorch(object):
    """Normalizes the targets of a dataset."""

    def __init__(self, means=None, stds=None):
        self.means = means
        self.stds = stds

    def fit(self, X):
        X = torch.tensor(X, dtype=torch.float)
        self.means = torch.mean(X, dim=0)
        # https://github.com/pytorch/pytorch/issues/29372
        self.stds = torch.std(X, dim=0, unbiased=False) + 1e-5

    def transform(self, X):
        X = torch.tensor(X, dtype=torch.float)
        return (X - self.means) / self.stds

    def inverse_transform(self, X):
        X = torch.tensor(X, dtype=torch.float)
        return X * self.stds + self.means

    def match_device(self, tensor):
        if self.means.device != tensor.device:
            self.means = self.means.to(tensor.device)
            self.stds = self.stds.to(tensor.device)

    def copy(self):
        return StandardScalerTorch(
            means=self.means.clone().detach(),
            stds=self.stds.clone().detach())

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"means: {self.means.tolist()}, "
            f"stds: {self.stds.tolist()})"
        )

def get_scaler_from_data_list(data_list, key):
    targets = torch.tensor([d[key] for d in data_list])
    scaler = StandardScalerTorch()
    scaler.fit(targets)
    return scaler