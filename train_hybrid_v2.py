"""Independent v2 bounded experiment; v1 files and checkpoints remain untouched."""
import argparse
import copy
import csv
import hashlib
import json
import random
from collections import deque
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
import train
from hybrid_control import HybridController as ReferenceController
from hybrid_v2 import ThreatController as HybridController, V2Config as ControllerConfig, SCHEMA as FEATURE_SCHEMA, ACTIONS as ACTION_SCHEMA, PPOV2 as HybridPPO, training_case
from hybrid_reward import SoftPathReward

from visual_state import VisualState

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'hybrid_experiment_v2'

def write_json(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8');temp.replace(path)

def append_json(path,value):
    with Path(path).open('a',encoding='utf-8') as handle:
        handle.write(json.dumps(value,ensure_ascii=False)+'\n')

def read_rows(path):
    if not Path(path).exists(): return []
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]

def make_env():
    scenario=train.scenarios.load('simple_tag.py').Scenario()
    return train.MultiAgentEnv(scenario.make_world(),scenario.reset_world,scenario.reward,
            scenario.observation,scenario.info,scenario.is_done)

def image_state(env,tracker,action=None):
    for viewer in env.viewers:
        if viewer is not None: viewer.window.switch_to()
    result=train.get_state(env,tracker,action)
    for viewer in env.viewers:
        if viewer is not None:
            viewer.window.set_vsync(False)
            viewer.window.flip=lambda:None
    return result

def scenarios(seed,count):
    rng=np.random.RandomState(seed);cases=[]
    obstacles=np.array([[-.35,.35],[.35,.35],[0.,-.35]])
    for _ in range(count):
        points=train.sample_positions(3,(-.8,.8),.25,rng,obstacles)
        cases.append(dict(agents=np.array(points[:2]).tolist(),goal=points[2].tolist(),obstacles=obstacles.tolist()))
    return cases

class Episode:
    def __init__(self,env,case,with_reward=True,reference=False):
        self.case=case;self.tracker=VisualState();self.steps=0
        self.parts={};self.trace=deque(maxlen=20);self.threat_steps=0;self.route_opportunities=0;self.route_choices=[0,0,0]
        env.reset(np.array(case['agents']),case['goal'],np.array(case['obstacles']))
        legacy=image_state(env,self.tracker)
        self.failed=legacy is None
        if self.failed: return
        self.controller=(ReferenceController if reference else HybridController)(self.tracker)
        self.obs=self.controller.prepare(self.tracker,legacy)
        self.reward=SoftPathReward(self.tracker) if with_reward else None

    def step(self,env,agent=None,training=False,direct=None):
        if direct is not None:
            action=direct.action(self.tracker).numpy()[0];route=0;command=np.zeros(2);sample=None
        elif agent is None:
            route=0;command=np.zeros(2);sample=None
            action=self.controller.execute(self.tracker,route,command)
        else:
            route,command,sample=agent.act(self.obs,training)
            action=self.controller.execute(self.tracker,route,command)
        telemetry=copy.deepcopy(getattr(self.controller,'telemetry',{}))
        self.threat_steps+=int(telemetry.get('threat',0)>.05)
        self.route_opportunities+=int(self.controller.decision)
        self.route_choices[int(route)]+=1
        action5=np.array([0.,action[0],0.,action[1],0.])
        _,raw,done,info=env.step(action5)
        self.steps+=1;outcome=info['n'][1]['outcome']
        legacy=image_state(env,self.tracker,action)
        if legacy is None: raise RuntimeError('Visual state unexpectedly lost after initialized map')
        next_obs=self.controller.prepare(self.tracker,legacy)
        parts=self.reward.reward(self.obs,next_obs,outcome,raw[1]) if self.reward else dict(raw=float(raw[1]))
        terminated=bool(any(done));truncated=not terminated and self.steps>=300
        if training: agent.record(sample,parts['total'],terminated,truncated,next_obs,action)
        for key,value in parts.items(): self.parts[key]=self.parts.get(key,0.)+value
        self.trace.append(dict(step=self.steps,obs=self.obs.tolist(),route=int(route),
            command=command.tolist(),executed=action.tolist(),reward=parts,prediction=telemetry))
        self.obs=next_obs
        return outcome or ('timeout' if truncated else None)

    def result(self,index,outcome):
        return dict(episode=index,outcome=outcome,steps=self.steps,
            switches=0 if self.failed else self.controller.switches,
            reward_parts=self.parts,last20=list(self.trace),sampling=self.case.get('sampling','regular'),
            threat_steps=self.threat_steps,route_opportunities=self.route_opportunities,route_choices=self.route_choices)

    def snapshot(self,env):
        physical=[]
        for a in env.world.agents:
            physical.append(dict(position=a.state.p_pos.copy(),velocity=a.state.p_vel.copy(),
                communication=a.state.c.copy(),u=a.action.u.copy(),c=a.action.c.copy()))
        return dict(episode=self,physical=physical)

    @staticmethod
    def restore(snapshot,env):
        ep=snapshot['episode'];case=ep.case
        env.reset(np.array(case['agents']),case['goal'],np.array(case['obstacles']))
        for agent,state in zip(env.world.agents,snapshot['physical']):
            agent.state.p_pos=state['position'].copy();agent.state.p_vel=state['velocity'].copy()
            agent.state.c=state['communication'].copy();agent.action.u=state['u'].copy();agent.action.c=state['c'].copy()
        return ep

def summarize(rows):
    n=len(rows);count={k:sum(r['outcome']==k for r in rows) for k in ('success','caught','obstacle_collision','timeout','detect_fail')}
    steps=[r['steps'] for r in rows if r['outcome']=='success']
    return dict(episodes=n,success_rate=count['success']/max(n,1),counts=count,
        avg_success_steps=float(np.mean(steps)) if steps else None,
        avg_switches=float(np.mean([r['switches'] for r in rows])) if rows else 0.)

def evaluate(env,cases,output,agent=None,direct=False,reference=False):
    rows=read_rows(output);completed={r['episode'] for r in rows}
    for index,case in enumerate(cases):
        if index in completed: continue
        ep=Episode(env,case,with_reward=False,reference=reference)
        outcome='detect_fail' if ep.failed else None
        if direct and not ep.failed:
            from evaluate_200 import VelocityController
            controller=VelocityController(ep.tracker,False)
        else: controller=None
        while outcome is None: outcome=ep.step(env,agent,False,controller)
        row=ep.result(index,outcome);append_json(output,row);rows.append(row)
        if (index+1)%25==0: print(Path(output).stem,index+1,flush=True)
    assert len(rows)==len(cases) and {r['episode'] for r in rows}==set(range(len(cases)))
    stats=summarize(rows);write_json(str(output)+'.summary.json',stats)
    print(Path(output).stem,json.dumps(stats),flush=True)
    return stats

def experiment(out=OUT,budget=65536,resume=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    config=dict(seed=15,budget=budget,rollout=2048,validation_interval=16384,
        gamma=.99,gae_lambda=.95,epochs=8,batch_size=256,lr=1e-4,
        gradient_clip=.5,kl=.03,entropy=.01,features=FEATURE_SCHEMA,
        actions=ACTION_SCHEMA,controller=asdict(ControllerConfig()),validation_seed=20261026,validation_count=100,
        holdout_seed=20261027,training_scenario_seed=15,hard_fraction_before16384=.25,hard_fraction_after16384=.5,selection='validation success rate; ties retain earliest')
    if (out/'config.json').exists():
        if json.loads((out/'config.json').read_text(encoding='utf-8'))!=config:
            raise ValueError('Existing experiment configuration differs')
        if not resume: raise ValueError('Experiment exists; use --resume or a new --output')
    else: write_json(out/'config.json',config)
    validation=scenarios(20261026,100);holdout=scenarios(20261027,200)
    old_cases=json.loads((ROOT/'evaluation_200/scenarios.json').read_text())['scenarios']
    for name,cases in (('validation',validation),('holdout',holdout)):
        write_json(out/(name+'_scenarios.json'),cases)
    manifest={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ('hybrid_v2.py','train_hybrid_v2.py','hybrid_control.py','hybrid_ppo.py','hybrid_reward.py','visual_state.py','train.py','multiagent/environment.py','multiagent/core.py','multiagent/scenarios/simple_tag.py')}
    if (out/'source_hashes.json').exists() and json.loads((out/'source_hashes.json').read_text())!=manifest: raise ValueError('Source changed since experiment started')
    write_json(out/'source_hashes.json',manifest)
    random.seed(15);np.random.seed(15);torch.manual_seed(15)
    agent=HybridPPO();env=make_env();eval_env=make_env()
    # Measure unlearned controller before training, on the same sets used later.
    baseline=evaluate(eval_env,old_cases,out/'base_original200.jsonl')
    base_validation=evaluate(eval_env,validation,out/'base_validation100.jsonl')
    reference_validation=evaluate(eval_env,validation,out/'reference_validation100.jsonl',reference=True)
    rng=np.random.RandomState(15);step=0;index=0;ep=None;best_rate=-1.;best_step=0
    if resume and (out/'latest.pt').exists():
        state=agent.load(out/'latest.pt',resume=True);step=state['step'];extra=state['extra']
        rng.set_state(extra['scenario_rng']);index=extra['episode_index']
        best_rate=extra['best_rate'];best_step=extra['best_step']
        if extra['runtime'] is not None: ep=Episode.restore(extra['runtime'],env)
        # Truncate records produced after the last durable checkpoint on recovery.
        for filename,key,limit in (('train_episodes.jsonl','episode',index),('updates.jsonl','step',step+1),('validation.jsonl','step',step+1)):
            path=out/filename
            if path.exists():
                records=[r for r in read_rows(path) if r[key]<limit]
                path.write_text(''.join(json.dumps(r)+'\n' for r in records),encoding='utf-8')
        print('Resuming at step',step,flush=True)
    def save(path):
        agent.save(path,step,dict(scenario_rng=rng.get_state(),episode_index=index,
            best_rate=best_rate,best_step=best_step,runtime=ep.snapshot(env) if ep else None))
    while step<budget:
        if ep is None:
            ep=Episode(env,training_case(rng,step))
            if ep.failed:
                append_json(out/'train_episodes.jsonl',ep.result(index,'detect_fail'));index+=1;ep=None;continue
        outcome=ep.step(env,agent,training=True);step+=1
        if outcome is not None:
            append_json(out/'train_episodes.jsonl',dict(ep.result(index,outcome),total_step=step))
            index+=1;ep=None
        if len(agent.buffer)>=2048 or step==budget:
            agent.update();append_json(out/'updates.jsonl',dict(agent.last_metrics,step=step))
            print('TRAIN',step,'/',budget,'episodes',index,json.dumps(agent.last_metrics),flush=True)
            if step%16384==0 or step==budget:
                checkpoint=out/f'checkpoint_{step}.pt'
                result=evaluate(eval_env,validation,out/f'validation_{step}.jsonl',agent)
                append_json(out/'validation.jsonl',dict(step=step,**result))
                if result['success_rate']>best_rate:
                    best_rate=result['success_rate'];best_step=step;save(out/'best.pt')
                save(checkpoint)
            save(out/'latest.pt')
    save(out/'final.pt')
    agent.load(out/'best.pt')
    mixed=evaluate(eval_env,old_cases,out/'hybrid_original200.jsonl',agent)
    base_hold=evaluate(eval_env,holdout,out/'base_holdout200.jsonl')
    hybrid_hold=evaluate(eval_env,holdout,out/'hybrid_holdout200.jsonl',agent)
    reference_hold=evaluate(eval_env,holdout,out/'reference_holdout200.jsonl',reference=True)
    base_rows={r['episode']:r for r in read_rows(out/'base_original200.jsonl')}
    mixed_rows={r['episode']:r for r in read_rows(out/'hybrid_original200.jsonl')}
    differences=np.array([int(mixed_rows[i]['outcome']=='success')-int(base_rows[i]['outcome']=='success') for i in range(200)])
    boot=np.random.RandomState(12345).choice(differences,(10000,200)).mean(axis=1)
    improvement=mixed['success_rate']-baseline['success_rate']
    selected_validation=next(r for r in read_rows(out/'validation.jsonl') if r['step']==best_step)
    validation_improvement=best_rate-base_validation['success_rate']
    gate=(best_rate>=.75 and validation_improvement>=.05-1e-12
        and selected_validation['counts']['obstacle_collision']/len(validation)<=.05)
    report=dict(budget=budget,completed_steps=step,train_seed=15,best_step=best_step,
        best_validation_success=best_rate,base_validation=base_validation,base_original=baseline,
        hybrid_original=mixed,base_holdout=base_hold,hybrid_holdout=hybrid_hold,reference_holdout=reference_hold,reference_validation=reference_validation,
        paired_success_difference=float(improvement),paired_bootstrap_ci95=np.quantile(boot,[.025,.975]).tolist(),
        original_paired_wins=int((differences==1).sum()),original_paired_losses=int((differences==-1).sum()),
        validation_improvement=float(validation_improvement),recommend_long_training=bool(gate),automatic_extension=False)
    for group in ('original200','holdout200'):
        left={r['episode']:r for r in read_rows(out/('base_'+group+'.jsonl'))}
        right={r['episode']:r for r in read_rows(out/('hybrid_'+group+'.jsonl'))}
        delta=np.array([int(right[i]['outcome']=='success')-int(left[i]['outcome']=='success') for i in range(200)])
        interval=np.quantile(np.random.RandomState(12345).choice(delta,(10000,200)).mean(1),[.025,.975]).tolist()
        report[group+'_paired']=dict(wins=int((delta==1).sum()),losses=int((delta==-1).sum()),difference=float(delta.mean()),ci95=interval)
    write_json(out/'report.json',report)
    report_markdown(out,report)
    print('EXPERIMENT COMPLETE',json.dumps(report),flush=True)

def report_markdown(out,report):
    lines=['# v2威胁感知混合PPO短训报告','',f"完成 {report['completed_steps']} 步；训练种子15；最佳验证检查点 {report['best_step']} 步。",'',
        '| 场景集 | 方法 | 成功率 | 被捕 | 撞障碍 | 超时 | 平均路线切换 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for key,label,method in (('base_original','原200局','新基础控制'),('hybrid_original','原200局','混合PPO'),
        ('reference_holdout','新留出200局','v1基础控制'),('base_holdout','留出200局','新基础控制'),('hybrid_holdout','留出200局','混合PPO')):
        r=report[key];c=r['counts']
        lines.append(f"| {label} | {method} | {r['success_rate']:.1%} | {c['caught']} | {c['obstacle_collision']} | {c['timeout']} | {r['avg_switches']:.2f} |")
    lines+=['',f"原200局配对成功率差：{report['paired_success_difference']:+.1%}，配对bootstrap 95%区间：{report['paired_bootstrap_ci95']}。",'',
        f"长训练门槛通过：{report['recommend_long_training']}。本轮没有自动延长训练。",'',
        '该结果来自一个训练种子，不代表跨种子稳定性。原200局用于对照；留出集未用于模型选择。',
        '检测失败另见JSON计数。逐局last20包括状态、路线选择、PPO指令、执行动作和奖励分项。']
    failures=[r for r in read_rows(Path(out)/'hybrid_original200.jsonl') if r['outcome']!='success']
    if failures:
        for kind in ('caught','obstacle_collision','timeout'):
            rows=[r for r in failures if r['outcome']==kind]
            if rows:
                ids=[r['episode'] for r in rows[:10]]
                lines.append(f"- {kind}：{len(rows)}局；示例场景编号 {ids}，可从逐局记录复核失败前20步。")
    (Path(out)/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--steps',type=int,default=65536);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.steps<1 or args.steps>65536: parser.error('Short-run budget must be between 1 and 65536')
    torch.set_num_threads(1)
    # Keep pickled runtime classes importable from external evaluation processes.
    from train_hybrid_v2 import experiment as run_experiment
    run_experiment(args.output,args.steps,args.resume)
