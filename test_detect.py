# -*- coding: utf-8 -*-
"""验证 detect.py 的检测精度：多组随机位置渲染 -> 检测 -> 与 ground truth 对比。"""
import numpy as np

import multiagent.scenarios as scenarios
from multiagent.environment import MultiAgentEnv
from detect import detect_positions


def sample_positions(n=6, bounds=(-0.7, 0.7), min_sep=0.15):
    """随机采样 n 个互不重叠的 2D 点。"""
    pts = []
    for _ in range(5000):  # 最多尝试 5000 次
        p = np.random.uniform(bounds[0], bounds[1], size=2)
        if all(np.linalg.norm(p - q) >= min_sep for q in pts):
            pts.append(p)
            if len(pts) == n:
                break
    return pts


def main():
    scenario = scenarios.load('simple_tag.py').Scenario()
    world = scenario.make_world()
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward,
                        scenario.observation, info_callback=None,
                        done_callback=scenario.is_done, shared_viewer=True)

    trials = 20
    errors = {'prey': [], 'predator': [], 'checkpoint': [], 'obstacles': []}

    for t in range(trials):
        pts = sample_positions(6)
        predator = pts[0]
        prey = pts[1]
        check = pts[2]
        obs = pts[3:6]

        env.reset(np.array([predator, prey]), check, obs)
        image = env.render('rgb_array')[0]

        res = detect_positions(image)

        gt = {
            'prey': prey,
            'predator': predator,
            'checkpoint': check,
        }
        for name, true in gt.items():
            pred = res[name]
            if pred is None:
                errors[name].append(np.inf)
            else:
                errors[name].append(np.linalg.norm(np.asarray(pred) - np.asarray(true)))

        # 障碍物：按最近邻匹配（顺序可能不同）
        pred_obs = res['obstacles']
        for true in obs:
            d = min(np.linalg.norm(np.asarray(p) - np.asarray(true)) for p in pred_obs) \
                if pred_obs else np.inf
            errors['obstacles'].append(d)

    print(f'=== 检测误差（世界坐标单位，共 {trials} 组随机场景） ===')
    for name, errs in errors.items():
        errs = np.array(errs, dtype=float)
        finite = errs[np.isfinite(errs)]
        miss = int(np.sum(~np.isfinite(errs)))
        print(f'{name:10s} mean={finite.mean():.5f}  max={finite.max():.5f}  miss={miss}/{len(errs)}')

    # 像素换算：1 世界单位 = 400 像素
    all_errs = np.concatenate([np.asarray(e, dtype=float) for e in errors.values()])
    all_errs = all_errs[np.isfinite(all_errs)]
    print('\n(1 世界单位 = 400 像素；总体 mean 误差约对应 '
          f'{all_errs.mean() * 400:.2f} 像素)')


if __name__ == '__main__':
    main()
