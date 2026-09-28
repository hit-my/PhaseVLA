"""Frozen pi05 AE, memory conditioning only during the early denoising segment."""
import dataclasses
import math
import torch
from openpi.models_pytorch.memory_ae import MemoryAEPytorch
from openpi.models_pytorch.futuremamba_config import FutureMambaPytorchConfig

def handoff_steps(ratio, steps):
    if steps <= 0 or not 0 <= ratio <= 1:
        raise ValueError('Invalid handoff schedule')
    return min(steps, math.ceil(ratio * steps))

def hybrid_chain(noise, steps, ratio, memory_velocity, plain_velocity):
    count = handoff_steps(ratio, steps)
    dt = torch.tensor(-1.0 / steps, device=noise.device, dtype=torch.float32)
    tau = torch.tensor(1.0, device=noise.device, dtype=torch.float32)
    x = noise
    for index in range(steps):
        velocity = memory_velocity if index < count else plain_velocity
        x = x + dt * velocity(x, tau.expand(x.shape[0]))
        tau = tau + dt
    return x

class FrozenMemoryAE(MemoryAEPytorch):
    def __init__(self, config):
        # Parent's zero handoff restriction describes the old, all-step variant.
        super().__init__(dataclasses.replace(config, handoff_ratio=0.0))
        values = {f.name:getattr(config,f.name) for f in dataclasses.fields(FutureMambaPytorchConfig)}
        values.update(handoff_ratio=0.4,architecture='frozen_memory_ae_early04_v1')
        self.config = FutureMambaPytorchConfig(**values)
        self.freeze_base()

    def freeze_base(self):
        self.base.requires_grad_(False)
        self.base.eval()

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def compute_query_loss(self, batch, *, noise=None, time=None):
        steps = int(self.config.num_denoise_steps)
        boundary = 1.0 - handoff_steps(self.config.handoff_ratio, steps) / steps
        if time is None:
            u = torch.distributions.Beta(1.5, 1.0).sample((len(batch.actions),)).to(batch.actions.device)
            time = boundary + (1 - boundary) * u
        if not bool(((time >= boundary) & (time <= 1)).all()):
            raise ValueError('Training time outside memory-owned high-noise segment')
        # Parent masks actions/noise before interpolation, freezes prefix only,
        # and keeps the differentiable path through AE operations into memory.
        return super().compute_query_loss(batch, noise=noise, time=time)

    @torch.no_grad()
    def sample_actions_with_memory(self, observation, history, executed_actions,
                                   executed_action_mask, noise=None, num_steps=None,
                                   handoff_ratio=None):
        steps = int(self.config.num_denoise_steps if num_steps is None else num_steps)
        ratio = self.config.handoff_ratio if handoff_ratio is None else float(handoff_ratio)
        handoff_steps(ratio, steps)
        batch_size = observation.state.shape[0]
        shape = (batch_size, self.config.action_horizon, self.config.action_dim)
        if noise is None:
            noise = self.base.sample_noise(shape, observation.state.device)
        if tuple(noise.shape) != shape:
            raise ValueError('Invalid action noise shape')
        token, history, diagnostics = self.futuremamba.advance_history(history, executed_actions, executed_action_mask)
        prefix = self.base.encode_frozen_prefix(observation, train=False)
        x = hybrid_chain(noise, steps, ratio,
            lambda x,t: self.memory_velocity(prefix.kv_cache, prefix.pad_mask, token, x, t),
            lambda x,t: self.base.denoise_step(observation.state, prefix.pad_mask, prefix.kv_cache, x, t))
        return x, history, dict(diagnostics, progress_calls=0, action_calls=steps,
                               handoff_steps=handoff_steps(ratio,steps),
                               memory_conditioned_calls=handoff_steps(ratio,steps))
