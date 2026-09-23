from __future__ import annotations

import json
import random
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .data import to_torch
from .normalization import normalize_inputs, delta8_from_norm, le_from_norm


def setup_ddp():
    rank=int(__import__('os').environ.get('RANK','0')); local=int(__import__('os').environ.get('LOCAL_RANK','0')); world=int(__import__('os').environ.get('WORLD_SIZE','1'))
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for training')
    torch.cuda.set_device(local); device=torch.device('cuda',local)
    if world>1: dist.init_process_group('nccl',init_method='env://')
    return rank,local,world,device


def barrier():
    if dist.is_available() and dist.is_initialized(): dist.barrier()


def raw_model(model): return model.module if isinstance(model,DDP) else model


def set_seed(seed:int,rank:int=0):
    random.seed(seed+rank); np.random.seed(seed+rank); torch.manual_seed(seed+rank); torch.cuda.manual_seed_all(seed+rank)


@torch.no_grad()
def predict_batch(model,batch_np,stats,device,amp=True,hard_gate=True,gate_threshold=0.5):
    tb=normalize_inputs(to_torch(batch_np,device),stats)
    with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=amp and device.type=='cuda'):
        out=model(tb['state_n'],tb['context_n'],tb['edge_n'],tb['mask'])
    d8=delta8_from_norm(out['delta8_norm'].float(),stats)
    pscaled=(raw_model(model).peeq_scaled_hard(out,gate_threshold) if hard_gate else raw_model(model).peeq_scaled_soft(out)).float()
    dpeeq=pscaled*float(stats['peeq_delta_scale'])
    state=tb['state'].float()
    next_state=torch.cat([state[:,:8]+d8, state[:,8:9]+dpeeq],dim=1)
    le=le_from_norm(out['le_norm'].float(),stats)
    gate=torch.sigmoid(out['plastic_logit'].float())
    return next_state,le,gate,out,tb


def save_checkpoint(path:Path,epoch:int,best:float,model,optimizer,scheduler,history:list,config:dict,stats:dict):
    path.parent.mkdir(parents=True,exist_ok=True)
    
    safe_cfg={k:(str(v) if isinstance(v,Path) else v) for k,v in config.items()}
    torch.save({'epoch':epoch,'best_val':best,'model':raw_model(model).state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'history':history,'config':safe_cfg,'stats':stats},path)


def load_model_checkpoint(path:Path,model,device,optimizer=None,scheduler=None):
    ck=torch.load(path,map_location=device)
    model.load_state_dict(ck['model'])
    if optimizer is not None and 'optimizer' in ck: optimizer.load_state_dict(ck['optimizer'])
    if scheduler is not None and 'scheduler' in ck: scheduler.load_state_dict(ck['scheduler'])
    return ck
