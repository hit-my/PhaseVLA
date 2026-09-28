"""Opt-in training-only Mamba-2 state updates. Never mutates incoming state.

Supports the deployed single-group, non-distributed Mamba-2 configuration.
This module is an isolated candidate fix, not a modification to old runs.
"""
import torch
from torch.nn import functional as F
from openpi.models_pytorch.mamba_memory import Mamba2MemoryBackend,MemorySnapshot

ORIGINAL_STEP=Mamba2MemoryBackend._step_impl

def mixer_step(mixer, hidden, conv_state, ssm_state):
    if mixer.ngroups != 1:
        raise ValueError('Differentiable candidate supports ngroups=1 only')
    projected=mixer.in_proj(hidden.squeeze(1));d_mlp=(projected.shape[-1]-2*mixer.d_ssm-2*mixer.d_state-mixer.nheads)//2
    z0,x0,z,xbc,dt=torch.split(projected,[d_mlp,d_mlp,mixer.d_ssm,mixer.d_ssm+2*mixer.d_state,mixer.nheads],dim=-1)
    next_conv=torch.cat((conv_state[:,:,1:],xbc[:,:,None].to(conv_state.dtype)),dim=-1)
    xbc=(next_conv*mixer.conv1d.weight.squeeze(1)).sum(dim=-1)
    if mixer.conv1d.bias is not None:xbc=xbc+mixer.conv1d.bias
    xbc=mixer.act(xbc).to(hidden.dtype)
    x,b,c=torch.split(xbc,[mixer.d_ssm,mixer.d_state,mixer.d_state],dim=-1)
    # Match the inference CUDA update's internal FP32 discretization.
    dt=F.softplus(dt.float()+mixer.dt_bias.float())
    decay=torch.exp(dt*(-torch.exp(mixer.A_log.float())))
    x=x.reshape(x.shape[0],mixer.nheads,mixer.headdim)
    contribution=torch.einsum('bh,bn,bhp->bhpn',dt,b.float(),x.float())
    updated=ssm_state.float()*decay[:,:,None,None]+contribution
    next_ssm=updated.to(ssm_state.dtype)
    y=torch.einsum('bhpn,bn->bhp',updated,c.float())
    d=mixer.D.float().reshape(1,mixer.nheads,mixer.headdim if mixer.D_has_hdim else 1)
    y=(y+d*x.float()).reshape(x.shape[0],mixer.d_ssm).to(hidden.dtype)
    y=Mamba2MemoryBackend._norm(mixer.norm,y,z) if mixer.rmsnorm else y*mixer.act(z)
    if d_mlp:y=torch.cat((F.silu(z0)*x0,y),dim=-1)
    return mixer.out_proj(y).unsqueeze(1),next_conv,next_ssm

def differentiable_step(self,x,state):
    if not torch.is_grad_enabled():
        return ORIGINAL_STEP(self,x,state)
    hidden=x[:,None];residual=None;layers=[]
    for block,layer_state in zip(self.layers,state.layers,strict=True):
        residual=hidden if residual is None else hidden+residual
        hidden=self._norm(block.norm,residual.to(block.norm.weight.dtype),None)
        if block.residual_in_fp32:residual=residual.float()
        hidden,conv,ssm=mixer_step(block.mixer,hidden,*layer_state)
        layers.append((conv,ssm))
    hidden=self._norm(self.norm,(hidden+residual).to(self.norm.weight.dtype),None)
    return hidden[:,0].to(x.dtype),MemorySnapshot(self.backend_id,self.state_schema_version,x.shape[0],tuple(layers))

def install_training_fix():
    Mamba2MemoryBackend._step_impl=differentiable_step
