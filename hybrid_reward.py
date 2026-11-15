"""Finite soft-cost distance potential, independent of policy route selection."""
from functools import lru_cache
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

@lru_cache(maxsize=1)
def grid():
    axis=np.linspace(-1.,1.,101)
    x,y=np.meshgrid(axis,axis,indexing='ij')
    points=np.stack((x,y),axis=-1)
    indexes=np.arange(10201).reshape(101,101)
    sources=[];targets=[];lengths=[]
    for dx,dy in ((1,0),(0,1),(1,1),(1,-1)):
        a=indexes[:101-dx,max(0,-dy):min(101,101-dy)].ravel()
        b=indexes[dx:,max(0,dy):min(101,101+dy)].ravel()
        sources.extend((a,b));targets.extend((b,a))
        lengths.extend((np.full(len(a),.02*np.hypot(dx,dy)),)*2)
    return points,np.concatenate(sources),np.concatenate(targets),np.concatenate(lengths)

class SoftPathReward:
    def __init__(self,tracker,gamma=.99,beta=2.):
        self.gamma=gamma;self.beta=beta
        self.goal=tracker.positions['checkpoint'].copy()
        self.obstacles=tracker.obstacles.copy()
        points,src,dst,length=grid()
        distance=np.linalg.norm(points[:,:,None,:]-self.obstacles,axis=-1).min(axis=-1)
        obstacle_cost=20*np.maximum(0.,1-distance/.09)**2
        outside=np.maximum(np.max(np.abs(points),axis=-1)-.825,0.)/.175
        costs=(1+obstacle_cost+20*outside**2).ravel()
        weight=length*(costs[src]+costs[dst])/2
        low=np.clip(np.floor((self.goal+1)/.02).astype(int),0,99)
        seeds=np.array([(low[0]+dx)*101+low[1]+dy for dx in (0,1) for dy in (0,1)])
        goal_cost=1+20*max(0.,1-np.min(np.linalg.norm(self.obstacles-self.goal,axis=1))/.09)**2
        connections=np.linalg.norm(points.reshape(-1,2)[seeds]-self.goal,axis=1)*(costs[seeds]+goal_cost)/2
        # The extra node is the exact image-derived goal, not a snapped source.
        src=np.concatenate((src,np.full(4,10201),seeds))
        dst=np.concatenate((dst,seeds,np.full(4,10201)))
        weight=np.concatenate((weight,connections,connections))
        graph=coo_matrix((weight,(src,dst)),shape=(10202,10202)).tocsr()
        self.field=dijkstra(graph,indices=10201,directed=True)[:10201].reshape(101,101)
        assert np.isfinite(self.field).all()

    def distance(self,point):
        point=np.asarray(point)
        position=np.clip((point+1)/.02,0,100)
        lower=np.minimum(np.floor(position).astype(int),99)
        u,v=position-lower;i,j=lower
        value=((1-u)*(1-v)*self.field[i,j]+u*(1-v)*self.field[i+1,j]
               +(1-u)*v*self.field[i,j+1]+u*v*self.field[i+1,j+1])
        return float(value+20*np.linalg.norm(point-np.clip(point,-1,1)))

    @staticmethod
    def risk_penalty(gap,closing,scale):
        return -.02*float(np.clip(closing/scale,0,1))*float(np.clip(1-gap/.08,0,1))**2

    def reward(self,previous,current,outcome=None,raw_reward=0.):
        before=-self.distance(previous[:2])
        after=0. if outcome is not None else -self.distance(current[:2])
        progress=self.beta*(self.gamma*after-before)
        terminal=30. if outcome=='success' else (-30. if outcome is not None else 0.)
        predator=0.
        if np.all(current[26:28]>.5):
            predator=self.risk_penalty(current[25]*2,current[23]*.53,.53)
        obstacle=0.
        risks=current[32:44].reshape(3,4)
        for index,row in enumerate(risks):
            if current[26]>.5 and current[29+index]>.5:
                obstacle=min(obstacle,self.risk_penalty(row[2]*2,row[1]*.25,.25))
        parts=dict(raw=float(raw_reward),progress=float(progress),terminal=terminal,
                   time=-.01,predator_risk=predator,obstacle_risk=obstacle)
        parts['total']=sum(parts[k] for k in parts if k!='raw')
        return parts
