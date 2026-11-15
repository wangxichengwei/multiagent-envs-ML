"""Independent v7 experiment: obstacle-safe 2D waypoint offset on the v3 rule."""
import argparse
import copy
import hashlib
import json
import random
from collections import deque
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

import train
from train_hybrid_v4 import (write_json, append_json, read_rows, make_env, image_state,
                             scenarios, summarize, paired)
from hybrid_v7 import V7Controller, PPOV7, V7Config, SCHEMA as FEATURE_SCHEMA, ACTIONS as ACTION_SCHEMA
from hybrid_v4 import EvasionReward
from hybrid_v2 import training_case
from visual_state import VisualState

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'hybrid_experiment_v7'


class Episode:
    def __init__(self, env, case, with_reward=True):
        self.case = case
        self.tracker = VisualState()
        self.steps = 0
        self.parts = {}
        self.trace = deque(maxlen=20)
        self.threat_steps = 0
        env.reset(np.array(case['agents']), case['goal'], np.array(case['obstacles']))
        legacy = image_state(env, self.tracker)
        self.failed = legacy is None
        if self.failed:
            return
        self.controller = V7Controller(self.tracker)
        self.obs = self.controller.prepare(self.tracker, legacy)
        self.reward = EvasionReward(self.tracker, evasion_reward_weight=V7Config().evasion_reward_weight) if with_reward else None

    def step(self, env, agent=None, training=False):
        if agent is None:
            command = np.zeros(2)
            sample = None
        else:
            command, sample = agent.act(self.obs, training)
        action = self.controller.execute(self.tracker, 0, command)
        rule_route = int(getattr(self.controller, 'rule_route', 0))
        telemetry = copy.deepcopy(getattr(self.controller, 'telemetry', {}))
        rule = copy.deepcopy(getattr(self.controller, 'rule_telemetry', {}))
        self.threat_steps += int(telemetry.get('threat', 0) > .05)
        action5 = np.array([0., action[0], 0., action[1], 0.])
        _, raw, done, info = env.step(action5)
        self.steps += 1
        outcome = info['n'][1]['outcome']
        legacy = image_state(env, self.tracker, action)
        if legacy is None:
            raise RuntimeError('Visual state unexpectedly lost after initialized map')
        next_obs = self.controller.prepare(self.tracker, legacy)
        parts = self.reward.reward(self.obs, next_obs, outcome, raw[1]) if self.reward else dict(raw=float(raw[1]))
        terminated = bool(any(done))
        truncated = not terminated and self.steps >= 300
        if training:
            agent.record(sample, parts['total'], terminated, truncated, next_obs, action)
        for key, value in parts.items():
            self.parts[key] = self.parts.get(key, 0.) + value
        self.trace.append(dict(step=self.steps, route_used=rule_route, command=command.tolist(),
                               executed=action.tolist(), reward=parts, prediction=telemetry, rule=rule))
        self.obs = next_obs
        return outcome or ('timeout' if truncated else None)

    def result(self, index, outcome):
        return dict(episode=index, outcome=outcome, steps=self.steps,
                    switches=0 if self.failed else self.controller.switches,
                    reward_parts=self.parts, last20=list(self.trace),
                    sampling=self.case.get('sampling', 'regular'), threat_steps=self.threat_steps)


def evaluate(env, cases, output, agent=None):
    rows = read_rows(output)
    completed = {r['episode'] for r in rows}
    for index, case in enumerate(cases):
        if index in completed:
            continue
        ep = Episode(env, case, with_reward=False)
        outcome = 'detect_fail' if ep.failed else None
        while outcome is None:
            outcome = ep.step(env, agent, False)
        row = ep.result(index, outcome)
        append_json(output, row)
        rows.append(row)
        if (index + 1) % 25 == 0:
            print(Path(output).stem, index + 1, flush=True)
    assert len(rows) == len(cases) and {r['episode'] for r in rows} == set(range(len(cases)))
    stats = summarize(rows)
    write_json(str(output) + '.summary.json', stats)
    print(Path(output).stem, json.dumps(stats), flush=True)
    return stats


def report_markdown(out, report):
    lines = ['# v7 障碍安全绕行点偏移 PPO 短训报告', '',
             'v7 让 PPO 学习二维绕行点偏移，但在使用前把偏移投影到安全集：目标点须在保守边界内，',
             '且小球到目标点的直线段须避开全部膨胀障碍物；不安全则二分收缩偏移。零偏移即 v3 规则。',
             '本实验不修改物理参数。', '']
    lines += ['| 场景集 | 方法 | 成功率 | 被捕 | 撞障碍 | 超时 | 检测失败 | 平均换路 |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for key, label, method in (
            ('rule_validation100', '验证100局', 'v3规则'),
            ('ppo_validation100', '验证100局', 'v7 PPO'),
            ('rule_original200', '原200局', 'v3规则'),
            ('ppo_original200', '原200局', 'v7 PPO'),
            ('rule_holdout200', '留出200局', 'v3规则'),
            ('ppo_holdout200', '留出200局', 'v7 PPO')):
        if key not in report:
            continue
        r = report[key]
        c = r['counts']
        lines.append(f"| {label} | {method} | {r['success_rate']:.1%} | {c['caught']} | "
                     f"{c['obstacle_collision']} | {c['timeout']} | {c['detect_fail']} | {r['avg_switches']:.2f} |")
    lines += ['']
    for group in ('original200', 'holdout200'):
        p = report.get(group + '_paired')
        if p:
            lines.append(f"{group} v7 PPO 相对 v3 规则的配对成功率差：{p['difference']:+.1%}；"
                         f"配对胜/负 {p['wins']}/{p['losses']}；95%配对bootstrap区间 {p['ci95']}。")
            lines.append(f"  - 新增成功场景：{p['wins_ids']}")
            lines.append(f"  - 丢失成功场景：{p['losses_ids']}")
    lines += ['', '该结果来自一个训练种子与固定配对场景，不代表跨种子稳定性。',
              '逐局记录包含每步 route_used、PPO 偏移指令、执行动作与奖励分项（含 predator_evasion）。']
    (out / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def experiment(out=OUT, budget=65536, resume=False):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    config = dict(seed=15, budget=budget, rollout=2048, validation_interval=16384,
                  gamma=.99, gae_lambda=.95, epochs=8, batch_size=256, lr=1e-4,
                  gradient_clip=.5, kl=.03, entropy=.01, features=FEATURE_SCHEMA,
                  actions=ACTION_SCHEMA, controller=asdict(V7Config()),
                  validation_seed=20261026, validation_count=100, holdout_seed=20261027,
                  selection='validation success rate; ties retain earliest')
    if (out / 'config.json').exists():
        if json.loads((out / 'config.json').read_text(encoding='utf-8')) != config:
            raise ValueError('Existing experiment configuration differs')
        if not resume:
            raise ValueError('Experiment exists; use --resume or a new --output')
    else:
        write_json(out / 'config.json', config)
    validation = scenarios(20261026, 100)
    holdout = scenarios(20261027, 200)
    original = json.loads((ROOT / 'evaluation_200' / 'scenarios.json').read_text(encoding='utf-8'))['scenarios']
    for name, cases in (('validation', validation), ('holdout', holdout)):
        write_json(out / (name + '_scenarios.json'), cases)
    manifest = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                for name in ('hybrid_v7.py', 'train_hybrid_v7.py', 'hybrid_v6.py', 'hybrid_v5.py',
                             'hybrid_v4.py', 'hybrid_v3.py', 'hybrid_v2.py', 'hybrid_control.py',
                             'hybrid_reward.py', 'visual_state.py', 'train.py',
                             'multiagent/environment.py', 'multiagent/core.py',
                             'multiagent/scenarios/simple_tag.py')}
    if (out / 'source_hashes.json').exists() and json.loads((out / 'source_hashes.json').read_text(encoding='utf-8')) != manifest:
        raise ValueError('Source changed since experiment started')
    write_json(out / 'source_hashes.json', manifest)
    random.seed(15)
    np.random.seed(15)
    torch.manual_seed(15)
    agent = PPOV7()
    env = make_env()
    eval_env = make_env()
    baseline_original = evaluate(eval_env, original, out / 'rule_original200.jsonl')
    baseline_validation = evaluate(eval_env, validation, out / 'rule_validation100.jsonl')
    rng = np.random.RandomState(15)
    step = 0
    index = 0
    ep = None
    best_rate = -1.
    best_step = 0
    if resume and (out / 'latest.pt').exists():
        state = agent.load(out / 'latest.pt', resume=True)
        step = state['step']
        rng.set_state(state['extra']['scenario_rng'])
        index = state['extra']['episode_index']
        best_rate = state['extra']['best_rate']
        best_step = state['extra']['best_step']
        for filename, key, limit in (('train_episodes.jsonl', 'episode', index),
                                     ('updates.jsonl', 'step', step + 1),
                                     ('validation.jsonl', 'step', step + 1)):
            path = out / filename
            if path.exists():
                records = [r for r in read_rows(path) if r[key] < limit]
                path.write_text(''.join(json.dumps(r) + '\n' for r in records), encoding='utf-8')
        print('Resuming at step', step, flush=True)

    def save(path):
        agent.save(path, step, dict(scenario_rng=rng.get_state(), episode_index=index,
                                    best_rate=best_rate, best_step=best_step))

    while step < budget:
        if ep is None:
            ep = Episode(env, training_case(rng, step))
            if ep.failed:
                append_json(out / 'train_episodes.jsonl', ep.result(index, 'detect_fail'))
                index += 1
                ep = None
                continue
        outcome = ep.step(env, agent, training=True)
        step += 1
        if outcome is not None:
            append_json(out / 'train_episodes.jsonl', dict(ep.result(index, outcome), total_step=step))
            index += 1
            ep = None
        if len(agent.buffer) >= 2048 or step == budget:
            agent.update()
            append_json(out / 'updates.jsonl', dict(agent.last_metrics, step=step))
            print('TRAIN', step, '/', budget, 'episodes', index, json.dumps(agent.last_metrics), flush=True)
            if step % 16384 == 0 or step == budget:
                result = evaluate(eval_env, validation, out / f'validation_{step}.jsonl', agent)
                append_json(out / 'validation.jsonl', dict(step=step, **result))
                if result['success_rate'] > best_rate:
                    best_rate = result['success_rate']
                    best_step = step
                    save(out / 'best.pt')
            save(out / 'latest.pt')
    save(out / 'final.pt')
    agent.load(out / 'best.pt')
    ppo_original = evaluate(eval_env, original, out / 'ppo_original200.jsonl', agent)
    rule_hold = evaluate(eval_env, holdout, out / 'rule_holdout200.jsonl')
    ppo_hold = evaluate(eval_env, holdout, out / 'ppo_holdout200.jsonl', agent)
    ppo_validation = next(r for r in read_rows(out / 'validation.jsonl') if r['step'] == best_step)
    report = dict(budget=budget, completed_steps=step, train_seed=15, best_step=best_step,
                  best_validation_success=best_rate,
                  rule_validation100=baseline_validation, ppo_validation100=ppo_validation,
                  rule_original200=baseline_original, ppo_original200=ppo_original,
                  rule_holdout200=rule_hold, ppo_holdout200=ppo_hold,
                  original200_paired=paired(read_rows(out / 'rule_original200.jsonl'),
                                            read_rows(out / 'ppo_original200.jsonl')),
                  holdout200_paired=paired(read_rows(out / 'rule_holdout200.jsonl'),
                                           read_rows(out / 'ppo_holdout200.jsonl')))
    write_json(out / 'report.json', report)
    report_markdown(out, report)
    print('EXPERIMENT COMPLETE', json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--steps', type=int, default=65536)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.steps < 1 or args.steps > 65536:
        parser.error('Short-run budget must be between 1 and 65536')
    torch.set_num_threads(1)
    from train_hybrid_v7 import experiment as run_experiment
    run_experiment(args.output, args.steps, args.resume)
