"""Masked categorical route + tanh Gaussian commands, with GAE PPO."""
import math
import random
from dataclasses import asdict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical, Normal
from hybrid_control import STATE_VERSION, FEATURE_SCHEMA, ControllerConfig

ACTION_SCHEMA={'route':['keep','left','right'],'continuous':['heading_offset_pi/2','speed_adjustment'],
               'transform':'tanh','interval':5}

def network(output):
    return nn.Sequential(nn.Linear(64,256),nn.ELU(),nn.Linear(256,64),nn.ELU(),
                         nn.Linear(64,32),nn.ELU(),nn.Linear(32,output))

class HybridActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor=network(5);self.critic=network(1)
        nn.init.zeros_(self.actor[-1].weight);nn.init.zeros_(self.actor[-1].bias)
        with torch.no_grad(): self.actor[-1].bias[0]=4.
        self.log_std=nn.Parameter(torch.full((2,),math.log(.15)))

    def distributions(self,obs):
        output=self.actor(obs)
        mask=obs[:,53:56]>.5
        routes=Categorical(logits=output[:,:3].masked_fill(~mask,-1e9))
        continuous=Normal(output[:,3:],self.log_std.clamp(math.log(.05),math.log(.5)).exp())
        return routes,continuous,self.critic(obs).reshape(-1)

    @staticmethod
    def corrected_logprob(distribution,latent):
        logdet=2*(math.log(2)-latent-F.softplus(-2*latent))
        return (distribution.log_prob(latent)-logdet).sum(-1)

    def evaluate(self,obs,route,latent,entropy=True):
        routes,continuous,value=self.distributions(obs)
        active=(obs[:,53:56]>.5).sum(-1)>1
        logprob=self.corrected_logprob(continuous,latent)+active*routes.log_prob(route)
        ent=None
        if entropy:
            ent=-self.corrected_logprob(continuous,continuous.rsample())+active*routes.entropy()
        return logprob,value,ent

class HybridPPO:
    def __init__(self,device='cpu'):
        self.device=torch.device(device)
        self.policy=HybridActorCritic().to(self.device)
        self.old_policy=HybridActorCritic().to(self.device)
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.optimizer=torch.optim.Adam(self.policy.parameters(),lr=1e-4)
        self.buffer=[];self.last_metrics={}

    def act(self,obs,training=False):
        tensor=torch.as_tensor(obs,dtype=torch.float32,device=self.device).unsqueeze(0)
        with torch.no_grad():
            routes,continuous,value=self.old_policy.distributions(tensor)
            route=routes.sample() if training else routes.logits.argmax(-1)
            latent=continuous.sample() if training else continuous.mean
            logprob,_,_=self.old_policy.evaluate(tensor,route,latent,entropy=False)
        sample=dict(obs=np.array(obs,copy=True),route=int(route.item()),
                    latent=latent[0].cpu().numpy(),logprob=float(logprob.item()),value=float(value.item()),
                    route_active=bool((tensor[0,53:56]>.5).sum()>1))
        return sample['route'],torch.tanh(latent)[0].cpu().numpy(),sample

    def record(self,sample,reward,terminated,truncated,next_obs,executed):
        tensor=torch.as_tensor(next_obs,dtype=torch.float32,device=self.device).unsqueeze(0)
        with torch.no_grad(): next_value=float(self.old_policy.critic(tensor).item())
        self.buffer.append(dict(sample,reward=float(reward),terminated=bool(terminated),
            truncated=bool(truncated),next_value=next_value,executed=np.array(executed,copy=True)))

    def gae(self):
        advantages=np.zeros(len(self.buffer),dtype=np.float32);last=0.
        for i in reversed(range(len(self.buffer))):
            row=self.buffer[i]
            delta=row['reward']+.99*(not row['terminated'])*row['next_value']-row['value']
            last=delta+.99*.95*(not(row['terminated'] or row['truncated']))*last
            advantages[i]=last
        values=np.array([r['value'] for r in self.buffer],dtype=np.float32)
        return advantages,advantages+values

    def update(self):
        if not self.buffer: return
        advantages,returns=self.gae()
        to_tensor=lambda value,dtype=torch.float32:torch.as_tensor(np.asarray(value),dtype=dtype,device=self.device)
        obs=to_tensor([r['obs'] for r in self.buffer]);route=to_tensor([r['route'] for r in self.buffer],torch.long)
        latent=to_tensor([r['latent'] for r in self.buffer]);old_log=to_tensor([r['logprob'] for r in self.buffer])
        adv=to_tensor(advantages);adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
        targets=to_tensor(returns);metrics=[];stop=False
        for _ in range(8):
            for ids in torch.randperm(len(obs),device=self.device).split(256):
                lp,value,entropy=self.policy.evaluate(obs[ids],route[ids],latent[ids])
                logratio=lp-old_log[ids];ratio=logratio.exp()
                kl=((ratio-1)-logratio).mean()
                if kl.item()>.03: stop=True;break
                policy_loss=-torch.minimum(ratio*adv[ids],ratio.clamp(.8,1.2)*adv[ids]).mean()
                value_loss=F.mse_loss(value,targets[ids])
                loss=policy_loss+.5*value_loss-.01*entropy.mean()
                self.optimizer.zero_grad();loss.backward()
                grad=nn.utils.clip_grad_norm_(self.policy.parameters(),.5)
                self.optimizer.step()
                with torch.no_grad(): self.policy.log_std.clamp_(math.log(.05),math.log(.5))
                metrics.append([loss.item(),value_loss.item(),entropy.mean().item(),kl.item(),float(grad)])
            if stop: break
        self.last_metrics=dict(samples=len(obs),minibatches=len(metrics),early_stop=stop)
        if metrics:
            self.last_metrics.update(zip(('loss','value_loss','entropy','kl','grad_norm'),np.mean(metrics,axis=0).tolist()))
        self.old_policy.load_state_dict(self.policy.state_dict());self.buffer.clear()

    def save(self,path,step,extra=None):
        if self.buffer: raise ValueError('Save only at completed rollout boundaries')
        state=dict(version=STATE_VERSION,features=FEATURE_SCHEMA,actions=ACTION_SCHEMA,
            controller=asdict(ControllerConfig()),step=step,policy=self.policy.state_dict(),
            optimizer=self.optimizer.state_dict(),torch_rng=torch.get_rng_state(),
            numpy_rng=np.random.get_state(),python_rng=random.getstate(),extra=extra or {})
        if torch.cuda.is_available(): state['cuda_rng']=torch.cuda.get_rng_state_all()
        # Replace only after a complete serialization, to survive interrupted writes.
        temporary=str(path)+'.tmp';torch.save(state,temporary)
        import os
        os.replace(temporary,path)

    def load(self,path,resume=False):
        # Own local checkpoints include Python/NumPy RNG tuples; never load untrusted files.
        state=torch.load(path,map_location=self.device,weights_only=False)
        if state.get('version')!=STATE_VERSION or state.get('features')!=FEATURE_SCHEMA or state.get('actions')!=ACTION_SCHEMA or state.get('controller')!=asdict(ControllerConfig()):
            raise ValueError('Incompatible hybrid checkpoint schema/configuration')
        self.policy.load_state_dict(state['policy']);self.old_policy.load_state_dict(state['policy'])
        self.buffer.clear()
        if resume:
            self.optimizer.load_state_dict(state['optimizer'])
            torch.set_rng_state(state['torch_rng'].cpu());np.random.set_state(state['numpy_rng']);random.setstate(state['python_rng'])
            if 'cuda_rng' in state and torch.cuda.is_available(): torch.cuda.set_rng_state_all(state['cuda_rng'])
        return state
