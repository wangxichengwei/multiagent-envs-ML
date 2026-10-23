# -*- coding: utf-8 -*-
"""扫描模型目录下所有 checkpoint，在固定初始条件集上评估成功率曲线。

判断"训练有没有让小球更会跑"，看这个脚本的输出，不看 training_reward.txt：
训练奖励混入了路径长度与时间惩罚，无法直接换算成成功率。

用法：
    python eval_sweep.py --model-path saved_models/Simple_Tag_PathGAE_v1 --episodes 30
    python eval_sweep.py --model-path saved_models/Simple_Tag_PathGAE_v1 --every 500 --episodes 50

每个 checkpoint 使用同一个 seed 重建初始条件，因此各点之间的成功率可比。
输出：终端表格 + <model-path>/eval_sweep.csv
"""
import argparse
import csv
import os
import re

import numpy as np

import train
from visual_state import VisualState


def list_checkpoints(model_path):
    """返回目录下所有 PPO_model_<ep> 的回合号，升序。"""
    eps = []
    for name in os.listdir(model_path):
        m = re.fullmatch(r'PPO_model_(\d+)', name)
        if m:
            eps.append(int(m.group(1)))
    return sorted(eps)


def evaluate(env, agent, cfg, episodes, seed):
    """在固定 seed 的初始条件集上跑 episodes 局，统计结局分布。"""
    rng = np.random.RandomState(seed)
    counts = dict(success=0, caught=0, obstacle_collision=0,
                  timeout=0, detect_fail=0)
    success_steps = []

    for _ in range(episodes):
        train.reset_env(env, cfg, rng)
        tracker = VisualState()
        state = train.get_state(env, tracker)
        if state is None:
            counts['detect_fail'] += 1
            continue

        for step in range(cfg.eval_steps):
            action = agent.select_action(state)          # train=False -> 确定性均值
            _, _, _, info = train._step_env(env, action, cfg)
            outcome = info['n'][1]['outcome']
            if outcome == 'success':
                counts['success'] += 1
                success_steps.append(step + 1)
                break
            if outcome is not None:                      # caught / obstacle_collision
                counts[outcome] += 1
                break
            nxt = train.get_state(env, tracker, action.detach().cpu().numpy())
            if nxt is None:
                counts['detect_fail'] += 1
                break
            state = nxt
        else:
            counts['timeout'] += 1                       # 300 步未结束

    return counts, success_steps


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-path', required=True, help='包含 PPO_model_* 的目录')
    p.add_argument('--episodes', type=int, default=30, help='每个 checkpoint 评估局数')
    p.add_argument('--every', type=int, default=200, help='只看回合号能被它整除的 checkpoint')
    p.add_argument('--seed', type=int, default=2026, help='评估初始条件种子')
    args = p.parse_args()

    cfg = train.Config()
    cfg.train = False
    cfg.model_path = os.path.join(args.model_path, '')
    env, agent = train.env_agent_config(cfg)

    all_eps = list_checkpoints(cfg.model_path)
    if not all_eps:
        print(f'{cfg.model_path} 下没有 checkpoint，先训练。')
        return
    eps = [e for e in all_eps if e % args.every == 0] or all_eps

    print(f'评估 {len(eps)} 个 checkpoint，每个 {args.episodes} 局，'
          f'初始条件 seed={args.seed}\n')
    header = f'{"ep":>7} {"成功率":>8} {"成功步数":>9} {"caught":>7} {"obstacle":>9} {"timeout":>8} {"detect":>7}'
    print(header)
    print('-' * len(header))

    rows = []
    for ep in eps:
        agent.load(cfg.model_path, ep)
        counts, steps = evaluate(env, agent, cfg, args.episodes, args.seed)
        n = sum(counts.values())
        rate = counts['success'] / max(n, 1)
        avg = float(np.mean(steps)) if steps else float('nan')
        rows.append(dict(episode=ep, episodes=n, success_rate=round(rate, 4),
                         avg_success_steps=round(avg, 2), **counts))
        print(f'{ep:7d} {rate:8.2%} {avg:9.1f} {counts["caught"]:7d} '
              f'{counts["obstacle_collision"]:9d} {counts["timeout"]:8d} '
              f'{counts["detect_fail"]:7d}')

    out = os.path.join(cfg.model_path, 'eval_sweep.csv')
    with open(out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'\n已写入 {out}')

    rates = [r['success_rate'] for r in rows]
    if len(rates) > 1:
        print(f'成功率: 首个 {rates[0]:.1%} -> 末个 {rates[-1]:.1%}，'
              f'峰值 {max(rates):.1%}，标准差 {np.std(rates):.1%}')
        if max(rates) - rates[-1] > 0.1:
            print('提示: 峰值明显高于末值，说明训练后段在退化，检查 KL / 熵 / 学习率。')


if __name__ == '__main__':
    main()
