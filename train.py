# -*- coding: utf-8 -*-
"""任务二：训练被追击者躲避追击者+障碍物，到达目标点（状态来自图像处理）。

流程：
  render("rgb_array") -> detect.detect_positions -> 归一化状态向量 -> PPO 训练。

状态：32维显式视觉状态（位置、速度、风险、障碍物身份、边界、上一动作和检测标记）。

关键点：
  - 每个 episode 用随机、互不重叠的初始位置重置（考核时位置是公布的，必须能泛化）。
  - 只控制被追击者（evader / agent[1]），追击者由 environment.step 内置脚本自动追捕。
  - 动作：连续方向 (x, y) ∈ [-1,1]^2，映射到 5 维动作的 [right/left, up/down] 分量。
  - 训练采用图像地图的路径势函数奖励；评估仍以环境成功率为准。
"""
import os
import json

import numpy as np
import torch
import cv2

import multiagent.scenarios as scenarios
from multiagent.environment import MultiAgentEnv
from agents.PPO.agent import PPO
from detect import detect_positions
from visual_state import VisualState, STATE_DIM
from path_reward import PathReward

# 训练时把渲染图缩小到 400x400 再做检测，检测速度约快 4 倍，精度仍足够
DETECT_SIZE = 400
DETECT_SCALE = DETECT_SIZE / 2.0  # 每世界单位像素数（世界范围为 [-1,1] 共 2 单位）

class Config:
    def __init__(self):
        self.algo = 'PPO'
        self.env_name = 'MULTIAGENT-SIMPLE-TAG'
        self.seed = 15
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 训练 / 评估开关
        self.train = True
        self.load_model = False
        self.model_path = 'saved_models/Simple_Tag_PathGAE_v1/'
        self.start_ep = 0
        os.makedirs(self.model_path, exist_ok=True)

        # 超参数
        self.lr = 1e-4
        self.gamma = 0.99
        self.eps_clip = 0.2
        self.K_epochs = 8
        self.rollout_steps = 2048
        self.gae_lambda = .95
        self.batch_size = 256
        self.entropy_coef = .01
        self.max_grad_norm = .5
        self.target_kl = .03
        self.path_beta = 2.
        self.risk_weight = 0.  # 先验证路径奖励，风险惩罚通过消融实验启用

        # 训练 / 评估轮次
        self.train_eps = 10000
        self.train_steps = 300       # 单 episode 最大步数
        self.eval_eps = 100
        self.eval_steps = 300

        # 训练中评估：每 eval_every 局用固定初始条件测一次成功率（0 表示关闭）。
        # 只做观测，不参与训练，也不写入 rollout buffer。
        self.eval_every = 200
        self.eval_episodes = 20

        # 动作 / 状态维度
        self.action_dim = 2
        self.state_dim = STATE_DIM  # 32

        # 重置随机化范围与最小间距（世界坐标，避免重叠导致初始化失败）
        self.reset_bounds = (-0.8, 0.8)
        self.reset_min_sep = 0.25
        self.obstacles = np.array([[-0.35, 0.35], [0.35, 0.35], [0.0, -0.35]])

        # 成功判定阈值（与 is_done 中到达打卡点阈值一致：agent.size+check.size=0.075）
        self.success_dist = 0.075


def sample_positions(n, bounds, min_sep, rng, obstacles=()):
    """采样 n 个互不重叠的 2D 点。"""
    pts = []
    for _ in range(5000):
        p = rng.uniform(bounds[0], bounds[1], size=2)
        if all(np.linalg.norm(p - q) >= min_sep for q in list(obstacles) + pts):
            pts.append(p)
            if len(pts) == n:
                return pts
    raise RuntimeError('无法采样足够的无重叠初始位置，请检查范围和最小间距')


def reset_env(env, cfg, rng):
    """固定考核障碍物，仅随机追击者、被追击者和目标点。"""
    pts = sample_positions(3, cfg.reset_bounds, cfg.reset_min_sep, rng, cfg.obstacles)
    predator = pts[0]
    prey = pts[1]
    check = pts[2]
    env.reset(np.array([predator, prey]), check, cfg.obstacles.copy())
    return prey, check


def get_state(env, tracker, previous_action=None):
    """渲染一帧 -> 检测 -> 带时序记忆的状态向量 (32 维)。"""
    image = env.render('rgb_array')[0]
    small = cv2.resize(image, (DETECT_SIZE, DETECT_SIZE),
                       interpolation=cv2.INTER_AREA)
    res = detect_positions(small, scale=DETECT_SCALE)
    return tracker.update(res, previous_action)


def _step_env(env, action, cfg):
    """执行一步：把 (1,2) 动作映射成 5 维动作并 step。"""
    act = torch.zeros(1, 5, device=cfg.device)
    act[0, 1] = action[0, 0]                     # x -> right/left
    act[0, 3] = action[0, 1]                     # y -> up/down
    return env.step(act.squeeze(0).cpu().numpy())


def quick_eval(env, agent, cfg, episodes, seed):
    """训练中评估：固定 seed 的初始条件，跑 episodes 局，返回结局计数与成功步数。

    关键点：评估期间必须关闭 agent.train，否则 select_action 会把确定性评估的
    状态与动作写进 rollout buffer，污染 on-policy 数据。
    """
    was_training = agent.train
    agent.train = False
    rng = np.random.RandomState(seed)
    horizon = getattr(cfg, 'eval_steps', 300)
    counts = dict(success=0, caught=0, obstacle_collision=0, timeout=0, detect_fail=0)
    success_steps = []
    try:
        for _ in range(episodes):
            reset_env(env, cfg, rng)
            tracker = VisualState()
            state = get_state(env, tracker)
            if state is None:
                counts['detect_fail'] += 1
                continue
            for step in range(horizon):
                action = agent.select_action(state)      # train=False -> 确定性均值
                _, _, _, info = _step_env(env, action, cfg)
                outcome = info['n'][1]['outcome']
                if outcome == 'success':
                    counts['success'] += 1
                    success_steps.append(step + 1)
                    break
                if outcome is not None:
                    counts[outcome] += 1
                    break
                nxt = get_state(env, tracker, action.detach().cpu().numpy())
                if nxt is None:
                    counts['detect_fail'] += 1
                    break
                state = nxt
            else:
                counts['timeout'] += 1                   # 300 步未结束
    finally:
        agent.train = was_training
    return counts, success_steps


def train(cfg, env, agent):
    print(f'Start train | env={cfg.env_name} algo={cfg.algo} device={cfg.device}')
    rng = np.random.RandomState(cfg.seed)
    rewards, ma_rewards = [], []
    training_rewards = []
    sampled_steps = 0
    eval_history = []
    eval_every = getattr(cfg, 'eval_every', 0)      # 0 = 关闭训练中评估
    eval_episodes = getattr(cfg, 'eval_episodes', 0)
    eval_log = cfg.model_path + 'eval_log.txt'
    if eval_every and not os.path.exists(eval_log):
        with open(eval_log, 'w', encoding='utf-8') as handle:
            handle.write('episode,success_rate,avg_success_steps,success,caught,'
                         'obstacle,timeout,detect_fail\n')

    def update_policy():
        agent.update()
        if agent.last_metrics:
            metrics = dict(agent.last_metrics, sampled_steps=sampled_steps)
            with open(cfg.model_path + 'ppo_metrics.jsonl', 'a', encoding='utf-8') as handle:
                handle.write(json.dumps(metrics) + '\n')

    for i_ep in range(cfg.start_ep, cfg.train_eps):
        reset_env(env, cfg, rng)
        tracker = VisualState()
        cur = get_state(env, tracker)
        if cur is None:
            continue  # 检测失败，跳过该 episode
        state = cur
        reward_model = PathReward(tracker, cfg.gamma, cfg.path_beta, cfg.risk_weight)

        ep_reward = 0.0
        ep_training_reward = 0.0
        for i_step in range(cfg.train_steps):
            action = agent.select_action(state)          # (1, action_dim)
            _, reward_n, done_n, info = _step_env(env, action, cfg)

            next_cur = get_state(env, tracker, action.detach().cpu().numpy())
            if next_cur is None:
                next_cur = cur  # 检测失败则沿用上一状态

            # 用路径势函数奖励替代环境原始距离惩罚；原始奖励仍单独记录。
            shaped = reward_model.reward(state, next_cur, info['n'][1]['outcome'])
            terminated = bool(any(done_n))
            truncated = not terminated and i_step + 1 == cfg.train_steps
            next_state = next_cur
            agent.record_result(shaped, terminated, truncated, next_state)
            sampled_steps += 1
            ep_training_reward += shaped
            if len(agent.buffer.rewards) >= cfg.rollout_steps:
                update_policy()

            ep_reward += reward_n[1]  # 记录原始奖励（观察用，不含塑形）

            cur = next_cur
            state = next_cur

            if True in done_n:
                break

        rewards.append(ep_reward)
        training_rewards.append(ep_training_reward)
        ma = 0.9 * ma_rewards[-1] + 0.1 * ep_reward if ma_rewards else ep_reward
        ma_rewards.append(ma)

        if (i_ep + 1) % 20 == 0:
            print(f'ep {i_ep+1:5d}/{cfg.train_eps}  reward={ep_reward:8.2f}  '
                  f'ma={ma:8.2f}  training_reward={ep_training_reward:8.2f}')
        if (i_ep + 1) % 200 == 0:
            agent.save(cfg.model_path, i_ep + 1)
            np.savetxt(cfg.model_path + 'reward.txt', rewards)
            np.savetxt(cfg.model_path + 'ma_reward.txt', ma_rewards)
            np.savetxt(cfg.model_path + 'training_reward.txt', training_rewards)

        # 训练中成功率评估：这是判断"有没有越练越会跑"的唯一直接证据。
        if eval_every and eval_episodes and (i_ep + 1) % eval_every == 0:
            counts, success_steps = quick_eval(env, agent, cfg, eval_episodes,
                                               cfg.seed + 2027)
            total = sum(counts.values())
            rate = counts['success'] / max(total, 1)
            avg_step = float(np.mean(success_steps)) if success_steps else float('nan')
            eval_history.append((i_ep + 1, rate, avg_step))
            print(f'  [eval] ep {i_ep+1:5d}  成功率 {rate:6.2%}  '
                  f'成功步数 {avg_step:5.1f}  timeout {counts["timeout"]:2d}  '
                  f'caught {counts["caught"]:2d}  obstacle {counts["obstacle_collision"]:2d}')
            with open(eval_log, 'a', encoding='utf-8') as handle:
                handle.write(f'{i_ep+1},{rate:.4f},{avg_step:.1f},{counts["success"]},'
                             f'{counts["caught"]},{counts["obstacle_collision"]},'
                             f'{counts["timeout"]},{counts["detect_fail"]}\n')

    if agent.buffer.rewards:
        update_policy()  # 更新最后不足 rollout_steps 的采样
    agent.save(cfg.model_path, cfg.train_eps)
    np.savetxt(cfg.model_path + 'reward.txt', rewards)
    np.savetxt(cfg.model_path + 'ma_reward.txt', ma_rewards)
    np.savetxt(cfg.model_path + 'training_reward.txt', training_rewards)
    if eval_history:
        np.savetxt(cfg.model_path + 'eval_success.txt', np.array(eval_history),
                   fmt=['%d', '%.4f', '%.2f'],
                   header='episode success_rate avg_success_steps', comments='')
    print('Training done.')
    return rewards, ma_rewards


def eval(cfg, env, agent):
    print('Start eval ...')
    rng = np.random.RandomState(cfg.seed + 999)
    rewards, success, steps_of_success = [], 0, []

    for i_ep in range(cfg.eval_eps):
        reset_env(env, cfg, rng)
        tracker = VisualState()
        cur = get_state(env, tracker)
        if cur is None:
            rewards.append(0.0)
            continue
        state = cur
        ep_reward = 0.0
        ok = False

        for i_step in range(cfg.eval_steps):
            action = agent.select_action(state)          # eval 模式返回确定性 mean
            _, reward_n, done_n, info = _step_env(env, action, cfg)
            ep_reward += reward_n[1]

            outcome = info['n'][1]['outcome']
            if outcome == 'success':
                ok = True
                steps_of_success.append(i_step + 1)
                break
            if True in done_n:
                break

            next_cur = get_state(env, tracker, action.detach().cpu().numpy())
            if next_cur is None:
                break
            cur = next_cur
            state = next_cur

        if ok:
            success += 1
        rewards.append(ep_reward)

    rate = success / cfg.eval_eps
    avg_step = np.mean(steps_of_success) if steps_of_success else float('nan')
    print(f'成功率: {success}/{cfg.eval_eps} = {rate:.2%}')
    print(f'成功平均步数: {avg_step:.1f}')
    print(f'平均 reward: {np.mean(rewards):.2f}')
    return rate


def env_agent_config(cfg):
    scenario = scenarios.load('simple_tag.py').Scenario()
    world = scenario.make_world()
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward,
                        scenario.observation, info_callback=scenario.info,
                        done_callback=scenario.is_done, shared_viewer=True)
    agent = PPO(cfg.state_dim, cfg.action_dim, cfg)
    return env, agent


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--train', action='store_true', help='训练模式')
    p.add_argument('--eval', action='store_true', help='评估模式')
    p.add_argument('--eps', type=int, default=None, help='训练 episode 数')
    p.add_argument('--steps', type=int, default=None, help='单 episode 最大步数')
    p.add_argument('--load-ep', type=int, default=None, help='加载第几个 checkpoint')
    p.add_argument('--model-path', default=None, help='模型目录；修复版默认使用独立目录')
    p.add_argument('--eval-every', type=int, default=None,
                   help='训练中每隔多少局评估一次成功率，0 表示关闭（默认 200）')
    p.add_argument('--eval-episodes', type=int, default=None,
                   help='训练中每次评估的局数（默认 20）')
    args = p.parse_args()

    cfg = Config()
    if args.model_path is not None:
        cfg.model_path = os.path.join(args.model_path, '')
        os.makedirs(cfg.model_path, exist_ok=True)
    if args.eps is not None:
        cfg.train_eps = args.eps
    if args.steps is not None:
        cfg.train_steps = args.steps
    if args.eval_every is not None:
        cfg.eval_every = args.eval_every
    if args.eval_episodes is not None:
        cfg.eval_episodes = args.eval_episodes
    if args.train:
        cfg.train = True
    elif args.eval:
        cfg.train = False

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    env, agent = env_agent_config(cfg)

    if cfg.train:
        if args.load_ep is not None:
            agent.load(cfg.model_path, args.load_ep)
            cfg.start_ep = args.load_ep
        train(cfg, env, agent)
    else:
        load_ep = args.load_ep if args.load_ep is not None else cfg.train_eps
        agent.load(cfg.model_path, load_ep)
        eval(cfg, env, agent)
