# -*- coding: utf-8 -*-
"""绘制训练奖励曲线（读取 saved_models/Simple_Tag_Detect/ 下的 reward.txt / ma_reward.txt）。"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无界面环境，直接存图
import matplotlib.pyplot as plt

REWARD_PATH = "saved_models/Simple_Tag_Detect/reward.txt"
MA_REWARD_PATH = "saved_models/Simple_Tag_Detect/ma_reward.txt"


def moving_average(x, w=20):
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode='same')


def main():
    reward = np.loadtxt(REWARD_PATH)
    ma_reward = np.loadtxt(MA_REWARD_PATH)

    plt.figure(figsize=(10, 5))
    plt.plot(reward, alpha=0.25, color='tab:blue', label='reward (raw)')
    plt.plot(moving_average(reward, 20), color='tab:blue',
             linewidth=1.5, label='reward (MA-20)')
    plt.plot(ma_reward, color='tab:red', linewidth=1.5, label='ma_reward (EMA)')

    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.title(f'PPO training reward ({len(reward)} episodes)')
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig('reward_curve.png', dpi=120)
    print(f'saved reward_curve.png ({len(reward)} episodes)')
    print(f'  best reward = {reward.max():.2f}')
    print(f'  last ma_reward = {ma_reward[-1]:.2f}')


if __name__ == '__main__':
    main()
