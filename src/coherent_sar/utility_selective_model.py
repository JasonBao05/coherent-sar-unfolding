"""Legacy candidate-router support shared by the revised split-phase model.

The revised SplitPhaseUtilityUnfolding256 overrides the forward path and
disables auxiliary supervision. Its public training pipeline uses only the
final four-term reconstruction objective. The ten-channel legacy router is
retained to preserve the original model initialization sequence.

Inference uses observations and reconstructed candidates only. Training labels
describe a local squared-error proxy, not a causal guarantee for later stages.
The always-on path exactly recovers the existing signed-DP architecture.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from .equivariant_selective_model import FixedMaskExchangeEquivariantSelectiveUnfolding256
from .components import local_coherence_magnitude, normalized_dc_residual
from .torch_complex import complex_to_channels


def wrapped(value):
    return torch.atan2(torch.sin(value),torch.cos(value))


class UtilityRouter(nn.Module):
    def __init__(self,channels=8,initial_probability=.99):
        super().__init__()
        if not 0<initial_probability<1:raise ValueError('Initial probability must be in (0,1)')
        self.network=nn.Sequential(nn.Conv2d(10,channels,3,padding=1),nn.SiLU(),nn.Conv2d(channels,1,3,padding=1))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.constant_(self.network[-1].bias,math.log(initial_probability/(1-initial_probability)))

    def forward(self,features):
        logits=self.network(features.detach())
        return torch.sigmoid(logits),logits


class UtilitySelectiveUnfolding256(FixedMaskExchangeEquivariantSelectiveUnfolding256):
    def __init__(self,*args,router_channels=8,initial_probability=.99,**kwargs):
        super().__init__(*args,**kwargs)
        self.utility_routers=nn.ModuleList(UtilityRouter(router_channels,initial_probability) for _ in range(self.stages))
        self._utility_records=[]

    def freeze_backbone(self):
        for name,p in self.named_parameters():p.requires_grad_(name.startswith('utility_routers.'))

    @staticmethod
    def _features(anchor,base1,base2,joint1,joint2,gb1,gb2,gj1,gj2):
        eps=1e-6
        gamma=local_coherence_magnitude(base1,base2,window_size=7,eps=1e-8)[:,0]
        rb1,rb2=normalized_dc_residual(gb1,1e-8)[:,0],normalized_dc_residual(gb2,1e-8)[:,0]
        d1,d2=joint1-base1,joint2-base2
        relative=((d1.abs()+d2.abs())/(base1.abs()+base2.abs()+eps)).clamp(0,2)/2
        def alignment(d,g):
            return (-(d*g.conj()).real/(d.abs()*g.abs()+eps)).clamp(-1,1)
        align=.5*(alignment(d1,gb1)+alignment(d2,gb2))
        phi0=torch.angle(base2*base1.conj());phi1=torch.angle(joint2*joint1.conj())
        phase_change=wrapped(phi1-phi0).abs()/math.pi
        amp0=(base1.abs()-base2.abs()).abs()/(base1.abs()+base2.abs()+eps)
        amp1=(joint1.abs()-joint2.abs()).abs()/(joint1.abs()+joint2.abs()+eps)
        rb=gb1.abs()+gb2.abs();rj=gj1.abs()+gj2.abs()
        residual_gain=((rb-rj)/(rb+rj+eps)).clamp(-1,1)
        return torch.stack((anchor[:,0],gamma,.5*(rb1+rb2),(rb1-rb2).abs(),relative,align,phase_change,amp0,amp1,residual_gain),dim=1)

    def forward(self,measurements1,measurements2,operator,*,mask1=None,mask2=None,return_diagnostics=False,coupling_override='selective'):
        if coupling_override not in ('selective','always_on','independent'):raise ValueError(coupling_override)
        batch=measurements1.shape[0]
        m1=self._mask_batch(mask1,batch,'mask1');m2=self._mask_batch(mask2,batch,'mask2')
        first=operator.adjoint_masked(measurements1,m1);second=operator.adjoint_masked(measurements2,m2)
        anchor=local_coherence_magnitude(first,second,window_size=7,eps=1e-8).detach()
        c1,c2=self.conditioner(m1),self.conditioner(m2)
        self._utility_records=[];stages=[]
        for stage in range(self.stages):
            pre=.35*F.softplus(self.raw_pre_steps[stage])
            dc1,_=self._data_consistency(first,measurements1,m1,operator,pre*(1+c1['pre_step_delta'][:,stage]))
            dc2,_=self._data_consistency(second,measurements2,m2,operator,pre*(1+c2['pre_step_delta'][:,stage]))
            reg=.12*torch.sigmoid(self.raw_regularization[stage])
            prior1=self._denoise(stage,dc1,reg*(1+c1['regularization_delta'][:,stage]),c1['gamma'][:,stage],c1['beta'][:,stage])
            prior2=self._denoise(stage,dc2,reg*(1+c2['regularization_delta'][:,stage]),c2['gamma'][:,stage],c2['beta'][:,stage])
            correction1,correction2=self._interaction(stage,prior1,prior2)
            strength=.08*torch.sigmoid(self.raw_interaction_scale[stage])
            scale=.5*(prior1.abs().mean((-2,-1))+prior2.abs().mean((-2,-1))).clamp_min(1e-4)
            coupled1=prior1+strength*scale[:,None,None]*correction1
            coupled2=prior2+strength*scale[:,None,None]*correction2
            post=.35*F.softplus(self.raw_post_steps[stage])
            step1=post*(1+c1['post_step_delta'][:,stage]);step2=post*(1+c2['post_step_delta'][:,stage])
            joint1,gj1=self._data_consistency(coupled1,measurements1,m1,operator,step1)
            joint2,gj2=self._data_consistency(coupled2,measurements2,m2,operator,step2)
            if coupling_override=='always_on':
                first,second=joint1,joint2
                if return_diagnostics:stages.append({'gate':torch.ones_like(anchor),'anchor_gamma':anchor})
                continue
            base1,gb1=self._data_consistency(prior1,measurements1,m1,operator,step1)
            base2,gb2=self._data_consistency(prior2,measurements2,m2,operator,step2)
            if coupling_override=='independent':
                first,second=base1,base2
                if return_diagnostics:stages.append({'gate':torch.zeros_like(anchor),'anchor_gamma':anchor})
                continue
            features=self._features(anchor,base1,base2,joint1,joint2,gb1,gb2,gj1,gj2)
            gate,logits=self.utility_routers[stage](features)
            first=base1+gate[:,0]*(joint1-base1)
            second=base2+gate[:,0]*(joint2-base2)
            if self.training:
                self._utility_records.append((logits,base1.detach(),base2.detach(),joint1.detach(),joint2.detach()))
            if return_diagnostics:
                stages.append({'gate':gate,'anchor_gamma':anchor,'features':features.detach()})
        outputs=[]
        for value,c in ((first,c1),(second,c2)):
            phase=torch.tanh(self.phase_refiner(complex_to_channels(value,channel_dim=1))[:,0])
            delta=.08*c['phase_scale'][:,None,None]*phase
            outputs.append(value*torch.exp(torch.complex(torch.zeros_like(delta),delta)))
        if return_diagnostics:return outputs[0],outputs[1],{'stages':stages,'coupling_override':coupling_override}
        return outputs[0],outputs[1]

    def utility_supervision_loss(self,truth1,truth2,weight,*,temperature=.1,loss_weights=None):
        if temperature<=0:raise ValueError('Temperature must be positive')
        if not self.training:raise RuntimeError('Utility labels are training-only')
        w=loss_weights or {'complex_weight':1.,'amplitude_weight':.35,'raw_phase_weight':.12,'differential_phase_weight':1.25}
        with torch.no_grad():
            power=(truth1.abs().square()+truth2.abs().square()).mean((-2,-1))[:,None,None].clamp_min(1e-8)
            reliability=weight/weight.mean((-2,-1))[:,None,None].clamp_min(1e-8)
            truthphase=torch.angle(truth2*truth1.conj())
            def energy(a,b):
                complex_error=((a-truth1).abs().square()+(b-truth2).abs().square())/power
                amplitude_error=((a.abs()-truth1.abs()).square()+(b.abs()-truth2.abs()).square())/power
                raw=.5*(wrapped(torch.angle(a)-torch.angle(truth1)).square()+wrapped(torch.angle(b)-torch.angle(truth2)).square())
                phase=wrapped(torch.angle(b*a.conj())-truthphase).square()
                return w['complex_weight']*complex_error+w['amplitude_weight']*amplitude_error+reliability*(w['raw_phase_weight']*raw+w['differential_phase_weight']*phase)
            labels=[]
            for logits,b1,b2,j1,j2 in self._utility_records:
                eb,ej=energy(b1,b2),energy(j1,j2)
                gain=((eb-ej)/(eb+ej+1e-8)).clamp(-1,1)
                labels.append((torch.sigmoid(gain/temperature)[:,None],gain.abs()[:,None]))
        if not labels:return self.raw_pre_steps.sum()*0
        losses=[]
        for (logits,*_), (target,confidence) in zip(self._utility_records,labels):
            loss=F.binary_cross_entropy_with_logits(logits,target,reduction='none')
            losses.append((loss*confidence).sum()/confidence.sum().clamp_min(1e-8))
        return torch.stack(losses).mean()
