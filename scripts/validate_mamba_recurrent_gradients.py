"""Validate Mamba recurrent outputs and derivatives against independent references.

Run with the pinned FutureMamba runtime and an available CUDA GPU.
This is a numerical audit, not a policy-success benchmark.
"""
import json
import torch
from openpi.models_pytorch.futuremamba_config import MambaMemoryConfig
from openpi.models_pytorch.mamba_memory import Mamba2MemoryBackend
torch.set_num_threads(2)
out = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name()}
# Independent differentiable sequence reference in float32, including parameter gradients.
torch.manual_seed(9);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
backend=Mamba2MemoryBackend(MambaMemoryConfig(d_model=128,depth=2,d_state=128,d_conv=4,expand=2,headdim=64,ngroups=1)).cuda().train()
inp=torch.randn(2,5,128,device='cuda',requires_grad=True);state=backend.initial_state(2,device=torch.device('cuda'),dtype=torch.float32)
initial=[s.clone() for layer in state.layers for s in layer];ys=[]
for t in range(5):
 y,state=backend.step(inp[:,t],state);ys.append(y)
recurrent=torch.stack(ys,1);reference=inp;residual=None
with torch.no_grad():
 native_state=backend.initial_state(2,device=torch.device('cuda'),dtype=torch.float32);native_outputs=[]
 for t in range(5):
  native_y,native_state=backend._step_impl(inp[:,t],native_state);native_outputs.append(native_y)
 native=torch.stack(native_outputs,1)
for block in backend.layers:reference,residual=block(reference,residual,inference_params=None)
reference=backend.norm((reference+residual).to(backend.norm.weight.dtype))
projection=torch.randn_like(reference)
params=[p for p in backend.parameters() if p.requires_grad]
gr=torch.autograd.grad((recurrent*projection).sum(),[inp]+params,retain_graph=True,allow_unused=True)
gs=torch.autograd.grad((reference*projection).sum(),[inp]+params,allow_unused=True)
out['fp32_reference']={'output_max_abs':float((recurrent-reference).abs().max()),'input_grad_max_abs':float((gr[0]-gs[0]).abs().max()),'parameter_grad_max_abs':max(float((a-b).abs().max()) for a,b in zip(gr[1:],gs[1:]) if a is not None and b is not None)}
out['fp32_reference']['candidate_vs_native_max_abs']=float((recurrent-native).abs().max())
out['fp32_reference']['native_vs_sequence_max_abs']=float((native-reference).abs().max())
print(json.dumps(out),flush=True)
assert torch.allclose(recurrent,native,atol=2e-5,rtol=2e-4),out['fp32_reference']
# Fused sequence kernels have a separate numerical error from the step kernel.
# Require candidate error no larger than the native-vs-sequence error, plus 2e-5.
assert (recurrent-reference).abs().max() <= (native-reference).abs().max()+2e-5
assert torch.allclose(gr[0],gs[0],atol=2e-4,rtol=2e-3),out['fp32_reference']
gradient_errors=[]
for (name,p),a,b in zip([(n,p) for n,p in backend.named_parameters() if p.requires_grad],gr[1:],gs[1:]):
 assert (a is None)==(b is None)
 if a is not None:
  gradient_errors.append({'name':name,'max_abs':float((a-b).abs().max()),'relative_l2':float((a-b).norm()/b.norm().clamp_min(1e-8)),'relative_max':float((a-b).abs().max()/b.abs().max().clamp_min(1e-8))})
out['parameter_gradient_errors']=gradient_errors
print(json.dumps(out),flush=True)
finite_checks=[]
def scalar_recurrent():
 st=backend.initial_state(2,device=torch.device('cuda'),dtype=torch.float32);vals=[]
 for pos in range(inp.shape[1]):
  val,st=backend.step(inp[:,pos],st);vals.append(val)
 return float((torch.stack(vals,1)*projection).double().sum().detach())
for (name,p),analytic,sequence_grad in zip([(n,p) for n,p in backend.named_parameters() if p.requires_grad],gr[1:],gs[1:]):
 if not name.endswith(('A_log','dt_bias')):continue
 original=p.detach().clone()
 for index in range(p.numel()):
  values={}
  try:
   for offset in [-2,-1,1,2]:
    with torch.no_grad():p.copy_(original);p.reshape(-1)[index]+=offset*.05
    values[offset]=scalar_recurrent()
  finally:
   with torch.no_grad():p.copy_(original)
  numerical=(values[-2]-8*values[-1]+8*values[1]-values[2])/(12*.05)
  predicted=float(analytic.reshape(-1)[index]);other=float(sequence_grad.reshape(-1)[index])
  finite_checks.append({'name':name,'index':index,'analytic':predicted,'finite_difference':numerical,'sequence_gradient':other,'absolute_error':abs(predicted-numerical)})
out['finite_difference_checks']=finite_checks
print(json.dumps(out),flush=True)
# Compare tensorwise errors: elementwise relative error is unstable at cancellations.
# Forward agreement is independently checked against the native inference kernel above.
assert all(e['relative_l2']<1e-3 and e['relative_max']<1e-3 for e in gradient_errors if not e['name'].endswith(('A_log','dt_bias'))),gradient_errors
# Discretization derivatives are independently checked against five-point differences.
assert all(e['absolute_error']<1e-4+5e-3*abs(e['analytic']) for e in finite_checks),finite_checks
# Check input state immutability after an update, not only the zero-state case.
before=[s.clone() for layer in state.layers for s in layer]
backend.step(torch.randn(2,128,device='cuda'),state)
out['incoming_state_unchanged']=all(torch.equal(a,b) for a,b in zip(before,[s for layer in state.layers for s in layer]))
assert out['incoming_state_unchanged']

out['passed']=True
print(json.dumps(out),flush=True)
