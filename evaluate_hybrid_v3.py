"""Standalone deterministic evaluation for the rule-based evasion controller.

Reuses the paired scenario generators from train_hybrid_v2 so the results are
directly comparable to the historical v2 experiment. Supported methods:

  --method base      v2 zero-instruction base controller
  --method reference v1 base controller (identical in zero-instruction mode)
  --method evasion   v3 rule-based evasion controller

`--compare` runs the full paired sets (validation100, original200, holdout200)
for base and evasion, plus reference on the holdout, and writes a paired report.
Nothing here starts training.
"""
import argparse
import copy
import json
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

import train_hybrid_v2 as experiment
from train_hybrid_v2 import OUT, make_env, scenarios, write_json, append_json, read_rows, summarize, image_state
from hybrid_control import HybridController as ReferenceController
from hybrid_v2 import ThreatController as BaseController
from hybrid_v3 import EvasionController

ROOT = Path(__file__).resolve().parent
OUT3 = ROOT / 'hybrid_experiment_v3'


def make_controller(kind, tracker):
    if kind == 'reference':
        return ReferenceController(tracker)
    if kind == 'evasion':
        return EvasionController(tracker)
    return BaseController(tracker)


class Episode:
    def __init__(self, env, case, kind):
        self.case = case
        self.tracker = experiment.VisualState()
        self.steps = 0
        self.trace = deque(maxlen=20)
        env.reset(np.array(case['agents']), case['goal'], np.array(case['obstacles']))
        legacy = image_state(env, self.tracker)
        self.failed = legacy is None
        if self.failed:
            return
        self.controller = make_controller(kind, self.tracker)
        self.obs = self.controller.prepare(self.tracker, legacy)

    def step(self, env):
        action = self.controller.execute(self.tracker, 0, np.zeros(2))
        rule_route = int(getattr(self.controller, 'rule_route', 0))
        telemetry = copy.deepcopy(getattr(self.controller, 'telemetry', {}))
        rule = copy.deepcopy(getattr(self.controller, 'rule_telemetry', {}))
        action5 = np.array([0., action[0], 0., action[1], 0.])
        _, _, done, info = env.step(action5)
        self.steps += 1
        outcome = info['n'][1]['outcome']
        legacy = image_state(env, self.tracker, action)
        if legacy is None:
            raise RuntimeError('Visual state unexpectedly lost after initialized map')
        self.obs = self.controller.prepare(self.tracker, legacy)
        self.trace.append(dict(step=self.steps, route_used=rule_route,
                               executed=action.tolist(), prediction=telemetry, rule=rule))
        return outcome

    def result(self, index, outcome):
        return dict(episode=index, outcome=outcome, steps=self.steps,
                    switches=0 if self.failed else self.controller.switches,
                    last20=list(self.trace))


def evaluate(env, cases, output, kind):
    rows = read_rows(output)
    completed = {r['episode'] for r in rows}
    for index, case in enumerate(cases):
        if index in completed:
            continue
        ep = Episode(env, case, kind)
        outcome = 'detect_fail' if ep.failed else None
        while outcome is None:
            outcome = ep.step(env)
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


def paired(left_rows, right_rows):
    by_ep = lambda rows: {r['episode']: r for r in rows}
    left, right = by_ep(left_rows), by_ep(right_rows)
    delta = np.array([int(right[i]['outcome'] == 'success') - int(left[i]['outcome'] == 'success')
                      for i in range(len(left))])
    interval = np.quantile(np.random.RandomState(12345).choice(delta, (10000, len(delta))).mean(1), [.025, .975]).tolist()
    return dict(difference=float(delta.mean()), wins=int((delta == 1).sum()),
                losses=int((delta == -1).sum()), ci95=interval,
                wins_ids=[int(i) for i in np.flatnonzero(delta == 1)],
                losses_ids=[int(i) for i in np.flatnonzero(delta == -1)])


def report_markdown(path, report):
    lines = ['# v3 规则式捕食者躲避评估报告', '',
             '规则式躲避控制器（`hybrid_v3.EvasionController`）在威胁预演判定直行会被截击时，',
             '自动切换到预演净间距更大的安全侧路线；安全场景与 v2 零指令基础控制器完全一致。',
             '本实验不训练、不修改物理参数。', '']
    lines += ['| 场景集 | 方法 | 成功率 | 被捕 | 撞障碍 | 超时 | 检测失败 | 平均换路 |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for key, label, method in (
            ('base_validation100', '验证100局', '基础'),
            ('evasion_validation100', '验证100局', '躲避'),
            ('base_original200', '原200局', '基础'),
            ('evasion_original200', '原200局', '躲避'),
            ('reference_holdout200', '留出200局', 'v1基础'),
            ('base_holdout200', '留出200局', '基础'),
            ('evasion_holdout200', '留出200局', '躲避')):
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
            lines.append(f"{group} 躲避相对基础的配对成功率差：{p['difference']:+.1%}；"
                         f"配对胜/负 {p['wins']}/{p['losses']}；95%配对bootstrap区间 {p['ci95']}。")
            lines.append(f"  - 新增成功场景：{p['wins_ids']}")
            lines.append(f"  - 丢失成功场景：{p['losses_ids']}")
    lines += ['', '该结果来自一个训练种子与固定配对场景，不代表跨种子稳定性。',
              '逐局记录包含每步选择路线、执行动作与威胁预演（rule 字段记录 keep/side 净间距与选择）。']
    Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8')


def compare(out_dir=OUT3):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    validation = scenarios(20261026, 100)
    holdout = scenarios(20261027, 200)
    original = json.loads((ROOT / 'evaluation_200' / 'scenarios.json').read_text(encoding='utf-8'))['scenarios']
    env = make_env()
    try:
        results = {
            'base_validation100': evaluate(env, validation, out_dir / 'base_validation100.jsonl', 'base'),
            'evasion_validation100': evaluate(env, validation, out_dir / 'evasion_validation100.jsonl', 'evasion'),
            'base_original200': evaluate(env, original, out_dir / 'base_original200.jsonl', 'base'),
            'evasion_original200': evaluate(env, original, out_dir / 'evasion_original200.jsonl', 'evasion'),
            'reference_holdout200': evaluate(env, holdout, out_dir / 'reference_holdout200.jsonl', 'reference'),
            'base_holdout200': evaluate(env, holdout, out_dir / 'base_holdout200.jsonl', 'base'),
            'evasion_holdout200': evaluate(env, holdout, out_dir / 'evasion_holdout200.jsonl', 'evasion'),
        }
    finally:
        for viewer in env.viewers:
            if viewer is not None and not getattr(viewer.window, 'has_exit', False):
                viewer.close()
    results['original200_paired'] = paired(read_rows(out_dir / 'base_original200.jsonl'),
                                           read_rows(out_dir / 'evasion_original200.jsonl'))
    results['holdout200_paired'] = paired(read_rows(out_dir / 'base_holdout200.jsonl'),
                                          read_rows(out_dir / 'evasion_holdout200.jsonl'))
    write_json(out_dir / 'report.json', results)
    report_markdown(out_dir / 'report.md', results)
    print(json.dumps(results, indent=2), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['base', 'reference', 'evasion'], default='evasion')
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--seed', type=int, default=1014)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--compare', action='store_true',
                        help='Run full paired comparison and write hybrid_experiment_v3/report.md')
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error('--episodes must be positive')
    torch.set_num_threads(1)
    if args.compare:
        compare()
        return
    output = args.output or OUT3 / 'manual_tests' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    cases = scenarios(args.seed, args.episodes)
    write_json(output / 'scenarios.json', dict(seed=args.seed, scenarios=cases))
    write_json(output / 'config.json', dict(method=args.method, episodes=args.episodes, seed=args.seed))
    env = make_env()
    try:
        evaluate(env, cases, output / 'episodes.jsonl', args.method)
    finally:
        for viewer in env.viewers:
            if viewer is not None and not getattr(viewer.window, 'has_exit', False):
                viewer.close()
    print('Results:', output, flush=True)


if __name__ == '__main__':
    main()
