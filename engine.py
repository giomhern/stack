import torch
from tqdm.auto import tqdm
from typing import Dict, List, Tuple 

def train_step(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader):
    return 