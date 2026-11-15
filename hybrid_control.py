"""Image-only geometry, turn-aware velocity control and 64D observation."""
from dataclasses import asdict, dataclass
import numpy as np
from scipy.sparse.csgraph import dijkstra
from visual_state import FEATURES as LEGACY_FEATURES, applied_action

STATE_VERSION = 'hybrid64-v1'
FEATURES = dict(LEGACY_FEATURES, obstacle_risk=slice(32,44),
    candidate_targets=slice(44,50), candidate_lengths=slice(50,53),
    candidate_mask=slice(53,56), turn=slice(56,57), base_speed=slice(57,58),
    route_age=slice(58,59), route_side=slice(59,62), previous_command=slice(62,64))
FEATURE_SCHEMA = {k:[v.start,v.stop] for k,v in FEATURES.items()}

@dataclass(frozen=True)
class ControllerConfig:
    radius: float = .10
    max_speed: float = .25
    brake_accel: float = .7
    gain: float = 8.
    hysteresis: float = .02
    interval: int = 5
    reached: float = .025

def risk_metrics(relative, relative_velocity, radius):
    """Unscaled gap, signed closing speed, predicted gap and TTC (1s sentinel)."""
    r,v = np.asarray(relative),np.asarray(relative_velocity)
    distance=float(np.linalg.norm(r)); vv=float(v@v)
    closing=-float(r@v)/max(distance,1e-9)
    t=float(np.clip(-float(r@v)/max(vv,1e-12),0,1))
    closest=max(0.,float(np.linalg.norm(r+t*v))-radius)
    ttc=1.
    if closing>0:
        c=float(r@r)-radius**2
        disc=float(r@v)**2-vv*c
        if c<=0: ttc=0.
        elif vv>1e-12 and disc>=0:
            entry=(-float(r@v)-np.sqrt(disc))/vv
            if 0<=entry<=1: ttc=float(entry)
    return np.array([max(0.,distance-radius),closing,closest,ttc])

class RouteGraph:
    def __init__(self, tracker, cfg):
        self.cfg=cfg
        self.goal=tracker.positions['checkpoint'].copy()
        self.obstacles=tracker.obstacles.copy()
        theta=np.arange(16)*2*np.pi/16
        offsets=np.column_stack((np.cos(theta),np.sin(theta)))*(cfg.radius/np.cos(np.pi/16)+1e-4)
        ring=(self.obstacles[:,None,:]+offsets).reshape(-1,2)
        self.nodes=np.vstack((self.goal,ring[np.max(np.abs(ring),axis=1)<=.825]))
        n=len(self.nodes)
        delta=self.nodes[None,:,:]-self.nodes[:,None,:]
        length=np.linalg.norm(delta,axis=-1)
        rel=self.obstacles[None,None,:,:]-self.nodes[:,None,None,:]
        fraction=np.clip(np.sum(rel*delta[:,:,None,:],axis=-1)/np.maximum(length[:,:,None]**2,1e-12),0,1)
        nearest=self.nodes[:,None,None,:]+fraction[:,:,:,None]*delta[:,:,None,:]
        clear=np.all(np.linalg.norm(nearest-self.obstacles,axis=-1)>=cfg.radius-1e-8,axis=-1)
        matrix=np.where(clear,length,np.inf);np.fill_diagonal(matrix,0.)
        self.cost,self.parent=dijkstra(matrix,directed=False,indices=0,return_predecessors=True)

    def visible(self, point, endpoints):
        endpoints=np.atleast_2d(endpoints);delta=endpoints-point
        rel=self.obstacles[None,:,:]-point
        t=np.clip(np.sum(rel*delta[:,None,:],axis=-1)/np.maximum(np.sum(delta**2,axis=-1)[:,None],1e-12),0,1)
        near=point+t[:,:,None]*delta[:,None,:]
        return np.all(np.linalg.norm(near-self.obstacles,axis=-1)>=self.cfg.radius-1e-8,axis=1)

    def chain(self, index):
        path=[]
        for _ in range(len(self.nodes)):
            path.append(self.nodes[index].copy())
            if index==0: return path
            index=int(self.parent[index])
            if index<0: break
        return None

    def prune(self, point, path):
        if not path: return None
        visibility=self.visible(point,np.array(path))
        indexes=np.flatnonzero(visibility)
        return [p.copy() for p in path[indexes[-1]:]] if len(indexes) else None

    def recovery(self, point):
        distances=np.linalg.norm(self.obstacles-point,axis=1)
        index=int(np.argmin(distances))
        if distances[index]<self.cfg.radius+1e-5:
            direction=point-self.obstacles[index]
            if np.linalg.norm(direction)<1e-8: direction=np.array([1.,0.])
            direction=direction/max(np.linalg.norm(direction),1e-9)
            return [self.obstacles[index]+direction*(self.cfg.radius+.015), self.goal.copy()]
        safe=np.clip(point,-.8,.8)
        return [safe,self.goal.copy()]

def path_length(point,path):
    return float(np.linalg.norm(np.diff(np.vstack((point,path)),axis=0),axis=1).sum())

class HybridController:
    def __init__(self, tracker, cfg=None):
        self.cfg=cfg or ControllerConfig()
        self.graph=RouteGraph(tracker,self.cfg)
        self.current=None
        self.side=0
        self.age=self.cfg.interval
        self.previous_command=np.zeros(2)
        self.switches=0
        self.candidates=None
        self.manual_route=False

    def prepare(self, tracker, legacy_state):
        p=tracker.positions['prey'];g=self.graph
        recovery=np.min(np.linalg.norm(g.obstacles-p,axis=1))<self.cfg.radius
        if self.manual_route and self.current:
            while len(self.current)>1 and np.linalg.norm(self.current[0]-p)<.035:
                self.current=self.current[1:]
            # Explicit detours retain the selected first waypoint even when
            # a direct goal ray becomes clear; otherwise PPO's choice vanishes.
            pruned=self.current if g.visible(p,np.array(self.current[:1]))[0] else None
            if len(self.current)==1: self.manual_route=False;self.side=0
        else:
            pruned=g.prune(p,self.current) if self.current else None
        visible=g.visible(p,g.nodes) & np.isfinite(g.cost)
        indexes=np.flatnonzero(visible)
        ranked=sorted(indexes,key=lambda i: np.linalg.norm(g.nodes[i]-p)+g.cost[i])
        best=g.chain(int(ranked[0])) if ranked else g.recovery(p)
        if recovery: self.current=g.recovery(p)
        elif pruned:
            # Keep a learned route until it is no longer feasible. Only switch
            # back to a direct goal segment once it is genuinely visible.
            self.current=[g.goal.copy()] if visible[0] and not self.manual_route else pruned
            if not self.manual_route and path_length(p,best)+self.cfg.hysteresis<path_length(p,self.current):
                self.current=best
        else: self.current=best
        self.candidates=[self.current,None,None]
        if not recovery and self.age>=self.cfg.interval:
            goal_delta=g.goal-p
            for i in ranked:
                # The first node is a genuine optional detour, not a path to
                # prune immediately back to the direct ray to the goal.
                path=g.chain(int(i))
                if path is None: continue
                if np.linalg.norm(path[0]-self.current[0])<.02: continue
                cross=goal_delta[0]*(path[0]-p)[1]-goal_delta[1]*(path[0]-p)[0]
                slot=1 if cross>1e-8 else (2 if cross< -1e-8 else 0)
                if slot and self.candidates[slot] is None:
                    # Do not propose virtually equal variants of the keep path.
                    self.candidates[slot]=path
        self.mask=np.array([x is not None for x in self.candidates],dtype=bool)
        self.decision=self.age>=self.cfg.interval and bool(self.mask[1:].any()) and not recovery
        if not self.decision: self.mask[1:]=False
        turn,speed=self.profile(p,self.current)
        obs_risks=[]
        for obstacle in g.obstacles:
            metrics=risk_metrics(obstacle-p,-tracker.velocities['prey'],.075)
            obs_risks.extend(metrics/np.array([2.,.25,2.,1.]))
        targets=[];lengths=[]
        for available,path in zip(self.mask,self.candidates):
            targets.extend((path[0]-p)/2 if available else [0.,0.])
            lengths.append(path_length(p,path)/4 if available else 0.)
        onehot=np.eye(3)[self.side]
        result=np.concatenate((legacy_state,obs_risks,targets,lengths,self.mask.astype(float),
            [turn/np.pi,speed/.25,min(self.age/self.cfg.interval,1.)],onehot,self.previous_command))
        assert result.shape==(64,)
        return np.clip(result,-1,1).astype(np.float32)

    def profile(self, position, path):
        delta=path[0]-position;distance=np.linalg.norm(delta)
        turn=0.
        if len(path)>1:
            next_delta=path[1]-path[0]
            turn=float(np.arccos(np.clip(np.dot(delta,next_delta)/max(distance*np.linalg.norm(next_delta),1e-9),-1,1)))
            corner=.25*max(.25,np.cos(turn/2)**2)
            limit=np.sqrt(corner**2+2*self.cfg.brake_accel*max(distance-self.cfg.reached,0.))
        else:
            limit=np.sqrt(2*self.cfg.brake_accel*max(distance-self.cfg.reached,0.))
        return turn,min(self.cfg.max_speed,float(limit))

    def execute(self, tracker, route=0, command=None):
        command=np.zeros(2) if command is None else np.clip(np.asarray(command),-1,1)
        if route and (not self.decision or not self.mask[route]):
            raise ValueError('Invalid route selection')
        if self.decision:
            if route:
                # Hysteresis is applied to automatic candidate tie-breaking;
                # an explicit PPO request is allowed to choose a longer route.
                self.current=self.candidates[route]
                self.manual_route=True
                self.side=route;self.switches+=1
            self.age=0
        _,speed=self.profile(tracker.positions['prey'],self.current)
        direction=self.current[0]-tracker.positions['prey']
        direction=direction/max(np.linalg.norm(direction),1e-9)
        angle=command[0]*np.pi/2
        c,s=np.cos(angle),np.sin(angle)
        direction=np.array([c*direction[0]-s*direction[1],s*direction[0]+c*direction[1]])
        target_speed=min(speed,max(0.,speed*(1+command[1])))
        action=applied_action(self.cfg.gain*(direction*target_speed-tracker.velocities['prey']))
        self.previous_command=command.copy();self.age+=1
        return action
