"""Threat-conditioned route proposals, pursuit previews and effective speed control."""
from dataclasses import asdict, dataclass
import math
import os
import random
import numpy as np
import torch
from torch import nn
from hybrid_control import HybridController, ControllerConfig, FEATURE_SCHEMA as V1_SCHEMA, path_length
from hybrid_ppo import HybridPPO, HybridActorCritic
from visual_state import applied_action

VERSION='hybrid76-v2'
SCHEMA=dict(V1_SCHEMA,candidate_pursuit=slice(64,76))
SCHEMA={k:([v.start,v.stop] if isinstance(v,slice) else v) for k,v in SCHEMA.items()}
ACTIONS={'route':['keep','left','right'],'continuous':['heading_offset_pi/2','braking_or_speed_recovery'],
         'transform':'tanh','interval':5,'threat_gate':True}

@dataclass(frozen=True)
class V2Config(ControllerConfig):
    horizon: int = 30
    threat_gap: float = .16
    acceleration_boost: float = .25
    route_risk_weight: float = 1.2

def unit_limit(vectors,limit):
    norm=np.linalg.norm(vectors,axis=-1,keepdims=True)
    return vectors*np.minimum(1.,limit/np.maximum(norm,1e-9))

class ThreatController(HybridController):
    def __init__(self,tracker,cfg=None):
        super().__init__(tracker,cfg or V2Config())
        self.threat=0.;self.telemetry={}

    def preview(self,tracker,paths):
        """Vectorized 3-second pursuit model using vision and known dynamics only.

        Stops risk accumulation at arrival; does not count capture after success.
        Contact forces are not simulated; own predicted obstacle contact is flagged.
        """
        n=len(paths);maxlen=max(map(len,paths))
        waypoints=np.zeros((n,maxlen,2));sizes=np.array(list(map(len,paths)))
        for i,path in enumerate(paths):
            waypoints[i,:len(path)]=path;waypoints[i,len(path):]=path[-1]
        p=np.tile(tracker.positions['prey'],(n,1));v=np.tile(tracker.velocities['prey'],(n,1))
        pred=np.tile(tracker.positions['predator'],(n,1));pv=np.tile(tracker.velocities['predator'],(n,1))
        gap=np.maximum(0,np.linalg.norm(pred-p,axis=1)-.05)
        closest_time=np.zeros(n);index=np.zeros(n,dtype=int);active=np.ones(n,dtype=bool)
        obs_gap=np.full(n,np.inf);first_caps=None
        for step in range(self.cfg.horizon):
            target=waypoints[np.arange(n),index]
            arrived=(np.linalg.norm(target-p,axis=1)<.035)&(index<sizes-1)
            index+=arrived.astype(int);target=waypoints[np.arange(n),index]
            delta=target-p;distance=np.linalg.norm(delta,axis=1)
            next_target=waypoints[np.arange(n),np.minimum(index+1,sizes-1)]
            following=next_target-target
            cosine=np.clip(np.sum(delta*following,axis=1)/np.maximum(distance*np.linalg.norm(following,axis=1),1e-9),-1,1)
            corner=.25*np.maximum(.25,(1+cosine)/2)
            cap=np.minimum(.25,np.sqrt(corner**2+1.4*np.maximum(distance-.025,0)))
            final=index==sizes-1
            cap[final]=np.minimum(.25,np.sqrt(1.4*np.maximum(distance[final]-.025,0)))
            if first_caps is None:first_caps=cap.copy()
            control=unit_limit(8*(delta/np.maximum(distance[:,None],1e-9)*cap[:,None]-v),1.)
            pursuit=(p-pred)/np.maximum(np.linalg.norm(p-pred,axis=1,keepdims=True),1e-9)
            new_v=unit_limit(v+.1*control,.25)
            new_pv=unit_limit(pv+.085*pursuit,.28)
            new_p=p+.1*new_v;new_pred=pred+.1*new_pv
            relative=pred-p;relative_delta=(new_pred-new_p)-relative
            t=np.clip(-np.sum(relative*relative_delta,axis=1)/np.maximum(np.sum(relative_delta**2,axis=1),1e-9),0,1)
            new_gap=np.maximum(0,np.linalg.norm(relative+t[:,None]*relative_delta,axis=1)-.05)
            better=active&(new_gap<gap)
            gap[better]=new_gap[better];closest_time[better]=(step+t[better])*.1
            new_obs=np.linalg.norm(new_p[:,None,:]-self.graph.obstacles,axis=-1).min(axis=1)-.075
            obs_gap=np.minimum(obs_gap,np.where(active,new_obs,np.inf))
            p=np.where(active[:,None],new_p,p);pred=np.where(active[:,None],new_pred,pred)
            v=np.where(active[:,None],new_v,v);pv=np.where(active[:,None],new_pv,pv)
            active &= np.linalg.norm(p-self.graph.goal,axis=1)>=.075
            if not active.any():break
        eta=np.array([path_length(tracker.positions['prey'],path)/.25 for path in paths])
        return np.column_stack((gap,closest_time,eta,first_caps,obs_gap))

    def proposal_pool(self,tracker):
        g=self.graph;p=tracker.positions['prey'];direction=g.goal-p
        direction/=max(np.linalg.norm(direction),1e-9)
        normal=np.array([-direction[1],direction[0]])
        available=np.flatnonzero(g.visible(p,g.nodes)&np.isfinite(g.cost))
        ranked=sorted(available,key=lambda i:np.linalg.norm(g.nodes[i]-p)+g.cost[i])
        pool=[self.current];sides=[0];counts={1:0,2:0}
        def add(path):
            first=path[0]-p;cross=direction[0]*first[1]-direction[1]*first[0]
            side=1 if cross>1e-6 else (2 if cross< -1e-6 else 0)
            if not side or np.linalg.norm(path[0]-self.current[0])<.03:return
            if any(np.linalg.norm(path[0]-old[0])<.03 for old in pool):return
            pool.append(path);sides.append(side)
        for i in ranked:
            path=g.chain(int(i));delta=path[0]-p
            cross=direction[0]*delta[1]-direction[1]*delta[0]
            side=1 if cross>1e-6 else (2 if cross< -1e-6 else 0)
            if side and counts[side]<3:
                before=len(pool);add(path)
                if len(pool)>before:counts[side]+=1
        # Open-field evasion points are available even with no blocking obstacle.
        for sign in (-1,1):
            for forward,lateral in ((.12,.20),(.25,.25),(-.05,.22)):
                waypoint=p+direction*forward+normal*lateral*sign
                if np.max(np.abs(waypoint))>.8 or not g.visible(p,[waypoint])[0]:continue
                visible=np.flatnonzero(g.visible(waypoint,g.nodes)&np.isfinite(g.cost))
                if not len(visible):continue
                i=min(visible,key=lambda k:np.linalg.norm(g.nodes[k]-waypoint)+g.cost[k])
                add([waypoint]+g.chain(int(i)))
        return pool,sides

    def prepare(self,tracker,legacy_state):
        prefix=super().prepare(tracker,legacy_state)
        recovery=np.min(np.linalg.norm(self.graph.obstacles-tracker.positions['prey'],axis=1))<self.cfg.radius
        predictions=self.preview(tracker,[self.current])
        self.threat=float(np.clip(1-predictions[0,0]/self.cfg.threat_gap,0,1))
        gate=self.threat>.05 and self.age>=self.cfg.interval and not recovery
        if gate:
            pool,sides=self.proposal_pool(tracker)
            predictions=self.preview(tracker,pool)
        else:pool,sides=[self.current],[0]
        self.candidates=[self.current,None,None];chosen=[0,None,None]
        if gate:
            for side in (1,2):
                eligible=[i for i,s in enumerate(sides) if s==side and predictions[i,4]>=.005]
                if eligible:
                    def score(i):
                        danger=max(0.,1-predictions[i,0]/self.cfg.threat_gap)**2
                        return path_length(tracker.positions['prey'],pool[i])+self.cfg.route_risk_weight*danger
                    index=min(eligible,key=score)
                    self.candidates[side]=pool[index];chosen[side]=index
        self.mask=np.array([p is not None for p in self.candidates],dtype=bool)
        self.decision=bool(gate and self.mask[1:].any())
        prefix[44:56]=0
        extra=np.zeros((3,4))
        for side,index in enumerate(chosen):
            if index is None:continue
            prefix[44+2*side:46+2*side]=(self.candidates[side][0]-tracker.positions['prey'])/2
            prefix[50+side]=path_length(tracker.positions['prey'],self.candidates[side])/4
            prefix[53+side]=1
            extra[side]=predictions[index,:4]/[2.,3.,20.,.25]
        _,cap=self.profile(tracker.positions['prey'],self.current)
        nominal=cap
        prefix[57]=nominal/.25
        self.telemetry=dict(threat=self.threat,nominal_speed=nominal,speed_cap=cap,
            candidates=[None if i is None else predictions[i].tolist() for i in chosen])
        return np.clip(np.concatenate((prefix,extra.ravel())),-1,1).astype(np.float32)

    @staticmethod
    def commanded_speed(cap,command):
        # Negative commands lower the setpoint. Positive commands improve
        # velocity recovery gain below, without reducing baseline cruise speed.
        return cap*(1+min(0.,command))

    def execute(self,tracker,route=0,command=None):
        command=np.zeros(2) if command is None else np.clip(np.asarray(command),-1,1)
        if route and (not self.decision or not self.mask[route]):raise ValueError('Invalid route')
        if self.decision:
            if route:
                self.current=self.candidates[route];self.manual_route=True;self.side=route;self.switches+=1
            self.age=0
        _,cap=self.profile(tracker.positions['prey'],self.current)
        effective=command if self.threat>.05 else np.zeros(2)
        speed=self.commanded_speed(cap,float(effective[1]))
        direction=self.current[0]-tracker.positions['prey'];direction/=max(np.linalg.norm(direction),1e-9)
        angle=float(effective[0])*np.pi/2;c,s=np.cos(angle),np.sin(angle)
        direction=np.array([c*direction[0]-s*direction[1],s*direction[0]+c*direction[1]])
        # Boost only the along-track recovery term. Additional gain on the
        # lateral term would accelerate cornering beyond the base controller.
        error=speed*direction-tracker.velocities['prey']
        along=max(0.,float(error@direction))
        boost=self.cfg.acceleration_boost*max(0.,float(effective[1]))*along*direction
        action=applied_action(self.cfg.gain*(error+boost))
        self.previous_command=command.copy();self.age+=1
        return action

class ActorCriticV2(HybridActorCritic):
    def __init__(self):
        nn.Module.__init__(self)
        def net(output):
            return nn.Sequential(nn.Linear(76,256),nn.ELU(),nn.Linear(256,64),nn.ELU(),
                                 nn.Linear(64,32),nn.ELU(),nn.Linear(32,output))
        self.actor=net(5);self.critic=net(1)
        nn.init.zeros_(self.actor[-1].weight);nn.init.zeros_(self.actor[-1].bias)
        with torch.no_grad():self.actor[-1].bias[0]=1.
        self.log_std=nn.Parameter(torch.full((2,),math.log(.15)))

    def evaluate(self,obs,route,latent,entropy=True):
        routes,continuous,value=self.distributions(obs)
        route_active=(obs[:,53:56]>.5).sum(-1)>1
        # Safe-scene commands are ignored by the controller; their random
        # likelihood must not add noise to the policy gradient or KL estimate.
        command_active=obs[:,64]*2 < .16*.95
        logprob=command_active*self.corrected_logprob(continuous,latent)+route_active*routes.log_prob(route)
        ent=None
        if entropy:
            ent=command_active*(-self.corrected_logprob(continuous,continuous.rsample()))+route_active*routes.entropy()
        return logprob,value,ent

class PPOV2(HybridPPO):
    def __init__(self,device='cpu'):
        self.device=torch.device(device)
        self.policy=ActorCriticV2().to(self.device);self.old_policy=ActorCriticV2().to(self.device)
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.optimizer=torch.optim.Adam(self.policy.parameters(),lr=1e-4)
        self.buffer=[];self.last_metrics={}

    def save(self,path,step,extra=None):
        if self.buffer:raise ValueError('Save only after update')
        state=dict(version=VERSION,features=SCHEMA,actions=ACTIONS,controller=asdict(V2Config()),
            policy=self.policy.state_dict(),optimizer=self.optimizer.state_dict(),step=step,
            torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),extra=extra or {})
        torch.save(state,str(path)+'.tmp');os.replace(str(path)+'.tmp',path)

    def load(self,path,resume=False):
        state=torch.load(path,map_location=self.device,weights_only=False)
        if state.get('version')!=VERSION or state.get('features')!=SCHEMA or state.get('actions')!=ACTIONS or state.get('controller')!=asdict(V2Config()):
            raise ValueError('Incompatible v2 checkpoint')
        self.policy.load_state_dict(state['policy']);self.old_policy.load_state_dict(state['policy']);self.buffer.clear()
        if resume:
            self.optimizer.load_state_dict(state['optimizer']);torch.set_rng_state(state['torch_rng'].cpu())
            np.random.set_state(state['numpy_rng']);random.setstate(state['python_rng'])
        return state

def training_case(rng,step):
    """Only generated training scenes; never mine validation/holdout failures."""
    from train import sample_positions
    obstacles=np.array([[-.35,.35],[.35,.35],[0.,-.35]])
    fraction=.25 if step<16384 else .5
    if rng.rand()<fraction:
        for _ in range(512):
            prey,goal=sample_positions(2,(-.8,.8),.25,rng,obstacles)
            delta=goal-prey;length=np.linalg.norm(delta)
            if length<.7:continue
            normal=np.array([-delta[1],delta[0]])/length
            pred=prey+rng.uniform(.2,.7)*delta+rng.uniform(-.12,.12)*normal
            if np.max(np.abs(pred))>.8:continue
            if min(np.linalg.norm(pred-prey),np.linalg.norm(pred-goal))<.3:continue
            if np.min(np.linalg.norm(obstacles-pred,axis=1))<.25:continue
            return dict(agents=[pred.tolist(),prey.tolist()],goal=goal.tolist(),obstacles=obstacles.tolist(),sampling='intercept')
    pts=sample_positions(3,(-.8,.8),.25,rng,obstacles)
    return dict(agents=np.array(pts[:2]).tolist(),goal=pts[2].tolist(),obstacles=obstacles.tolist(),sampling='regular')
