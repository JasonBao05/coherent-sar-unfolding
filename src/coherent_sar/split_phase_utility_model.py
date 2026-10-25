"""Revised DP-JMRNet common/differential phase routing, keeping joint amplitudes.

This spatial router controls the tradeoff between single-image
complex accuracy and differential phase. All-one and full-independent overrides
recover the old paths. Zero *phase* routing keeps joint amplitudes and therefore
must not be described as removing all cross-epoch interaction.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .utility_selective_model import UtilitySelectiveUnfolding256,wrapped
from .components import local_coherence_magnitude
from .torch_complex import complex_to_channels


def split_phase_deltas(base1,base2,joint1,joint2):
    d1=torch.angle(joint1*base1.conj())
    d2=torch.angle(joint2*base2.conj())
    # Principal per-epoch increments; algebra reconstructs both endpoints modulo2pi.
    # The chosen half-angle convention is frozen and symmetric under epoch exchange.
    return .5*(d1+d2),.5*(d2-d1)


def route_split_phases(base1,base2,joint1,joint2,gate):
    common,differential=split_phase_deltas(base1,base2,joint1,joint2)
    dc=(gate[:,0]-1)*common
    dd=(gate[:,1]-1)*differential
    first=joint1*torch.exp(torch.complex(torch.zeros_like(dc),dc-dd))
    second=joint2*torch.exp(torch.complex(torch.zeros_like(dc),dc+dd))
    return first,second


class SplitPhaseRouter(nn.Module):
    def __init__(self,channels=8,initial_probability=.5):
        super().__init__()
        if not 0<initial_probability<1:raise ValueError(initial_probability)
        self.network=nn.Sequential(nn.Conv2d(12,channels,3,padding=1),nn.SiLU(),nn.Conv2d(channels,2,3,padding=1))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.constant_(self.network[-1].bias,math.log(initial_probability/(1-initial_probability)))
    def forward(self,features):
        logits=self.network(features.detach())
        return torch.sigmoid(logits),logits


class SplitPhaseUtilityUnfolding256(UtilitySelectiveUnfolding256):
    def __init__(self,*args,router_channels=8,initial_probability=.5,**kwargs):
        kwargs.setdefault("signed_phase_features", True)
        super().__init__(*args,router_channels=router_channels,initial_probability=initial_probability,**kwargs)
        self.utility_routers=nn.ModuleList(SplitPhaseRouter(router_channels,initial_probability) for _ in range(self.stages))

    def utility_supervision_loss(self,*args,**kwargs):
        # Round3 uses only the unchanged final reconstruction objective.
        return self.raw_pre_steps.sum()*0

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
                if return_diagnostics:stages.append({'gate':torch.ones_like(anchor).expand(-1,2,-1,-1),'anchor_gamma':anchor})
                continue
            base1,gb1=self._data_consistency(prior1,measurements1,m1,operator,step1)
            base2,gb2=self._data_consistency(prior2,measurements2,m2,operator,step2)
            if coupling_override=='independent':
                first,second=base1,base2
                if return_diagnostics:stages.append({'gate':torch.zeros_like(anchor).expand(-1,2,-1,-1),'anchor_gamma':anchor})
                continue
            features=self._features(anchor,base1,base2,joint1,joint2,gb1,gb2,gj1,gj2)
            common,differential=split_phase_deltas(base1,base2,joint1,joint2)
            features=torch.cat((features,common.abs()[:,None]/math.pi,differential.abs()[:,None]/math.pi),dim=1)
            gate,logits=self.utility_routers[stage](features)
            first,second=route_split_phases(base1,base2,joint1,joint2,gate)
            if self.training:self._utility_records.append((logits,base1.detach(),base2.detach(),joint1.detach(),joint2.detach()))
            if return_diagnostics:stages.append({'gate':gate,'anchor_gamma':anchor,'features':features.detach(),'common_phase_increment':common,'differential_half_increment':differential})
        outputs=[]
        for value,c in ((first,c1),(second,c2)):
            phase=torch.tanh(self.phase_refiner(complex_to_channels(value,channel_dim=1))[:,0])
            delta=.08*c['phase_scale'][:,None,None]*phase
            outputs.append(value*torch.exp(torch.complex(torch.zeros_like(delta),delta)))
        if return_diagnostics:return outputs[0],outputs[1],{'stages':stages,'coupling_override':coupling_override}
        return outputs[0],outputs[1]
