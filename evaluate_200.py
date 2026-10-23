"""Reproducible image-only comparison; resumable per-episode records."""
import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import train
from path_reward import PathReward
from visual_state import VisualState, applied_action

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'evaluation_200'
METHODS = ['ppo_3400','ppo_4000','ppo_5200','ppo_7400','ppo_10000','velocity_direct','velocity_path']

class VelocityController:
    def __init__(self, tracker, path):
        self.path = PathReward(tracker) if path else None
        self.last_target = None

    def action(self, tracker):
        position = tracker.positions['prey']
        target = tracker.positions['checkpoint']
        if self.path and not self.path.visible(position,target):
            choices = [(np.linalg.norm(position-node)+cost,i) for i,(node,cost) in enumerate(zip(self.path.nodes,self.path.costs))
                       if np.isfinite(cost) and self.path.visible(position,node)]
            if choices:
                best,index = min(choices)
                if self.last_target is not None:
                    old = self.last_target
                    if self.path.visible(position,self.path.nodes[old]):
                        old_cost = np.linalg.norm(position-self.path.nodes[old])+self.path.costs[old]
                        if old_cost <= best+.02: index=old
                self.last_target=index
                target=self.path.nodes[index]
        delta=target-position
        distance=np.linalg.norm(delta)
        desired_speed=min(.25, math.sqrt(2*.7*max(distance-.025,0)))
        desired_velocity=delta/max(distance,1e-9)*desired_speed
        action=applied_action(8*(desired_velocity-tracker.velocities['prey']))
        return torch.as_tensor(action,dtype=torch.float32).reshape(1,2)

def worker(method):
    torch.set_num_threads(1)
    cfg=train.Config(); cfg.train=False
    env,agent=train.env_agent_config(cfg)
    if method.startswith('ppo_'): agent.load(cfg.model_path,int(method.split('_')[1]))
    scenarios=json.loads((OUT/'scenarios.json').read_text())['scenarios']
    output=OUT/(method+'.jsonl')
    completed={r['episode'] for r in read_rows(output)}
    with output.open('a',encoding='utf-8',buffering=1) as handle:
        for index,case in enumerate(scenarios):
            if index in completed: continue
            env.reset(np.array(case['agents']),case['goal'],np.array(case['obstacles']))
            tracker=VisualState(); state=train.get_state(env,tracker)
            for viewer in env.viewers:
                if viewer is not None:
                    viewer.window.set_vsync(False)
                    viewer.window.flip = lambda: None
            controller=VelocityController(tracker,method=='velocity_path') if state is not None and method.startswith('velocity_') else None
            outcome='detect_fail' if state is None else 'timeout'
            steps=0
            if state is not None:
                for step in range(300):
                    action=controller.action(tracker) if controller else agent.select_action(state)
                    _,_,_,info=train._step_env(env,action,cfg)
                    steps=step+1
                    result=info['n'][1]['outcome']
                    if result is not None:
                        outcome=result; break
                    state=train.get_state(env,tracker,action.numpy())
                    if state is None: outcome='detect_fail'; break
            handle.write(json.dumps(dict(method=method,episode=index,outcome=outcome,steps=steps))+'\n')
            if (index+1)%20==0: print(method,index+1,flush=True)
    print(method,'finished',flush=True)

def read_rows(path):
    if not path.exists(): return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def summarize():
    summary=[]
    for method in METHODS:
        rows=read_rows(OUT/(method+'.jsonl'))
        if not rows: continue
        n=len(rows); k=sum(r['outcome']=='success' for r in rows); p=k/n
        z=1.96; center=(p+z*z/(2*n))/(1+z*z/n)
        half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
        success_steps=[r['steps'] for r in rows if r['outcome']=='success']
        row=dict(method=method,episodes=n,success=k,success_rate=p,ci_low=center-half,ci_high=center+half,
                 caught=sum(r['outcome']=='caught' for r in rows),obstacle=sum(r['outcome']=='obstacle_collision' for r in rows),
                 timeout=sum(r['outcome']=='timeout' for r in rows),detect_fail=sum(r['outcome']=='detect_fail' for r in rows),
                 avg_success_steps=float(np.mean(success_steps)) if success_steps else None)
        summary.append(row)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    if summary:
        with (OUT/'summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=list(summary[0])); writer.writeheader();writer.writerows(summary)
    return summary

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker');parser.add_argument('--summary',action='store_true');args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.summary: print(json.dumps(summarize(),indent=2)); return
    if args.worker: worker(args.worker);return
    if not (OUT/'scenarios.json').exists():
        cfg=train.Config();rng=np.random.RandomState(20261023);cases=[]
        for _ in range(200):
            pts=train.sample_positions(3,cfg.reset_bounds,cfg.reset_min_sep,rng,cfg.obstacles)
            cases.append(dict(agents=np.array(pts[:2]).tolist(),goal=pts[2].tolist(),obstacles=cfg.obstacles.tolist()))
        (OUT/'scenarios.json').write_text(json.dumps(dict(seed=20261023,horizon=300,scenarios=cases),indent=2))
    def launch(method):
        with (OUT/(method+'.log')).open('a') as log:
            result=subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--worker',method],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        print(method,'exit',result.returncode,flush=True)
        if result.returncode: raise RuntimeError(method+' failed; see its log')
    with ThreadPoolExecutor(max_workers=7) as pool: list(pool.map(launch,METHODS))
    print(json.dumps(summarize(),indent=2),flush=True)

if __name__=='__main__':main()
