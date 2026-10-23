# 环境配置

## 学习链路修复版

推荐使用 `train.py`。修复版在设置双方动作后推进物理，重置时清空动作；
固定三个考核障碍物，仅随机双方和目标点。环境的双方 done 表示同一回合结束，
具体原因从 `info['n'][1]['outcome']` 读取：`success`、`caught`、
`obstacle_collision`，未结束为 `None`。碰撞与到达同一步发生时碰撞优先。
超时由训练循环单独记录，不改成环境的真实终止。

PPO 对超时回合使用末状态价值自举，回报不跨回合传播；价值目标保留原奖励尺度，
只标准化优势。加载模型同步两个策略网络，训练结束前更新剩余采样。
当前版本采用路径塑形奖励和GAE PPO。环境原始奖励和运动参数保持原设置，
训练循环单独计算学习奖励；策略输出层及探索参数已改变，需要重新训练。
默认保存至 `saved_models/Simple_Tag_PathGAE_v1/`，旧模型保留。

```powershell
conda activate multiagent
python -m unittest test_learning_fixes test_visual_state test_path_ppo -v
python train.py --train --eps 10000
python train.py --eval --load-ep 10000
# 从第2000回合模型继续，训练到总计10000回合
python train.py --train --load-ep 2000 --eps 10000
```

使用 `--model-path` 指定其他模型目录。恢复训练会恢复网络和优化器、接续回合编号，
但不会恢复随机数状态和历史奖励曲线；当前运行的奖励文件只包含本次运行数据。

### 训练中成功率评估

`train.py` 每 `--eval-every` 局（默认 200）用固定初始条件的 `--eval-episodes` 局
（默认 20）评估当前策略，打印并追加到 `eval_log.txt`：

```text
episode,success_rate,avg_success_steps,success,caught,obstacle,timeout,detect_fail
```

训练结束汇总为 `eval_success.txt`（回合、成功率、成功平均步数）。评估期间把
`agent.train` 置为 False，既取确定性动作，也不把评估数据写入 rollout buffer。
`--eval-every 0` 关闭。评估会额外消耗时间，长跑可用
`--eval-every 500 --eval-episodes 10` 折中。

成功率是判断"有没有越练越会跑"的直接依据，`reward.txt` 不能替代它：原始奖励
混入了每步距离惩罚，塑形奖励又含有势差，都无法直接换算成成功率。
另需同时看失败类型分布——`timeout` 上升而 `caught` 下降，说明策略学会了
"躲着不动"而不是"逃到目标"，这是本任务最常见的局部最优。

### 显式视觉状态（32维）

`visual_state.py` 的 `VisualState` 在每局创建一次，训练和评估使用相同编码器。
不再拼接两帧。各位置均由 `detect_positions` 从图像提取，不读取仿真真实位置或速度。
时间步0.1秒、速度上限和边界几何是已知环境参数。

| 索引（从0开始） | 特征 | 缩放 |
| --- | --- | --- |
| 0–1 | 小球绝对位置 | 世界坐标 |
| 2–3 | 小球估计速度 | 除以0.25 |
| 4–5 | 捕食者相对位置 | 除以2 |
| 6–7 | 捕食者相对速度 | 除以0.53 |
| 8–9 | 终点相对位置 | 除以2 |
| 10–15 | 三个固定身份的障碍物相对位置 | 除以2 |
| 16–19 | 左、右、下、上边界保守净空 | 除以2，可为负 |
| 20–21 | 上一步实际输入方向 | 与环境相同的单位范数限幅 |
| 22–25 | 捕食者净间距、接近速度、最近时刻、预计最近净间距 | 距离除以2，速度除以0.53，时刻除以1秒 |
| 26–31 | 本帧检测有效标记：小球、捕食者、终点、障碍物0/1/2 | 0或1 |

速度由检测位置差除以两次检测的实际时间间隔计算，再做指数平滑（新估计权重0.6）；
首帧速度为0。风险按相对匀速运动预测未来1秒，不保证未来无碰撞，也不修改奖励或动作。
边界净空使用球心安全范围±0.825，作为对圆形边界阵列的保守近似。
所有特征最终限制在[-1,1]。

首次完整识别三个障碍物后，按x、y几何顺序编号；之后进行一对一最小距离匹配，
匹配距离阈值0.08。固定地图保留首次图像测得的中心，避免遮挡引起质心漂移。
障碍物或终点漏检时保留历史位置。移动物体漏检最多外推3步，之后保持位置、速度置0，
对应有效标记始终为0；重新检测后按真实间隔估计速度。每局重建编码器以清空历史。
首次地图未建立时返回无状态，训练跳过该局，评估计入失败。

输入从24维改为32维，旧模型不能直接加载，请在新的默认目录重新训练。

### 路径进展奖励与GAE PPO

每局从初始视觉地图建立可见性图：障碍物膨胀至半径0.09
（障碍物0.05＋小球0.025＋余量0.015），以16边外切多边形上的节点绕行。
图的边逐条检查是否穿过任何膨胀圆，使用Dijkstra计算到目标的可行路径距离。
这是静态障碍物的几何路径近似，不预测捕食者，也不直接控制动作。
球心保守边界为±0.825；目标到达半径0.075。
落入膨胀区、越出保守边界或无路可达时，距离返回有限上限4。

学习奖励为：

```text
r = terminal_reward - 0.01 + beta * (gamma * Phi(next) - Phi(current)) + risk
Phi = -可行路径距离
terminal_reward = 到达 +30；被捕/撞障碍物 -30；其他 0
beta = 2；gamma = 0.99
```

真实终止的下一势函数置0；超时保留下一势函数并自举。
合理绕行即使短暂远离目标，也能因路径距离缩短而获得更好的进展信号。
该势函数含折扣项，不能把单步奖励的正负直接解释为距离是否缩短。
可选风险惩罚仅在双方本帧检测有效、正在接近且预计最近净间距小于0.12时触发；
`Config.risk_weight=0` 默认关闭，便于先验证路径奖励的独立效果。

PPO 默认每2048步更新，可在回合中间更新；每次8轮、随机小批量256，
GAE lambda=0.95，clip=0.2，学习率1e-4，熵系数0.01，梯度范数裁剪0.5，
近似KL超过0.03时提前停止本次更新。GAE在真实终止和超时处停止跨局递推，
真实终止不自举，超时和采样批次末尾使用末状态价值。价值目标保留奖励单位，
只标准化优势。

策略学习高斯标准差（范围0.05–1.0，初值0.5）。高斯样本z通过
`a=z/sqrt(1+||z||^2)` 可逆映射到单位动作圆内，避免环境再次进行范数限幅；
策略概率含变换的Jacobian修正，熵由变换后分布的可重参数化采样估计。
评估使用变换后的高斯均值确定性动作，不添加探索噪声。

`reward.txt` 和 `ma_reward.txt` 仍记录环境原始奖励，
`training_reward.txt` 记录新学习奖励；`ppo_metrics.jsonl` 记录每次更新的损失、
价值误差、熵、KL、裁剪前梯度范数和小批量数。效果仍需以固定测试集成功率判断。
原32维模型缺少新的探索参数，不能直接恢复到本版，请使用新的默认目录重新训练。

质点小球追逃环境

> 参考 [博客](https://blog.csdn.net/kysguqfxfr/article/details/100070584?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-3.channel_param&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-3.channel_param)

## 安装

系统：ubuntu （windows下有些依赖可能会有问题）

依赖:

- gymnasium 
- pyglet = 1.5.24
- pytorch or tensorflow
- 可选：opencv-python (处理图像)

> 运行 `demo.py` 检查环境，缺什么包下载什么包.

## 环境介绍

> 以`demo.py`为例子。

首先创建并初始化环境：

```py
    scenario = scenarios.load("simple_tag.py").Scenario()
    world = scenario.make_world()
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None, done_callback=scenario.is_done, shared_viewer = True)
    
    env.reset()
    image = env.render("rgb_array")  # 使用这种方法读取图片
```

动作为五维向量，范围分别是`[0,1]`, 控制的是加速度方向和大小

```py
	# [noop, move right, move left, move up, move down]
        act_n = np.array([0,1,0,0,1])
```

状态为`(800,800,3)`大小的rgb图片，可以用于目标检测得到位置状态或者直接端到端训练智能体。整个画⾯的实际尺度为 [-1,1] * [-1,1] , 屏幕正中⼼为原点。画⾯宽⾼为800*800像素

```py
    image = env.render("rgb_array")  # 使用这种方法读取图片
```

另外，除了图像数据，也直接获取状态，获取方法与一般的环境一样

```py
obs_n = env.reset()
.
.
.
next_obs_n, reward_n, done_n, _ = env.step(act_n, 1)
```

在 simple_tag.py ⽂件中的环境类中的函数`observation()`有直接获取位置等信息的代码

但是这种⽅式是不符合⽼师所提出的要求的，⽼师是想让同学们⽤图像处理等知识获取状态信息。

障碍物位置`[-0.35, 0.35], [0.35, 0.35], [0, -0.35]`, 半径0.05.

重置函数, 参数为追击者位置 `[0,0]`，猎物位置 `[0.7, 0.7]`，目标点位置 `[-0.5, -0.5]`，障碍物位置 `obs_pos`。

```py
    env.reset(np.array([[0.0, 0.0], [0.7, 0.7]]), [-0.5, -0.5], obs_pos)
```

**重要**：重置时要注意不要重叠，否则会初始化失败

撞倒障碍物（不包括边界）和被追击者追上就会结束

```py
if True in done_n:
            break
```

## 环境文件

最重要的环境⽂件是 core.py , simple_tag.py , environment.py 这三个⽂件。这三个⽂件的⼤致调⽤关系是：core被simple_tag 调⽤，simple_tag被environment调⽤。
- core⽂件中主要声明了环境中出现的各个实体：agent，border，landmark，check等。
- simple_tag⽂件主要设置实体的参数（初始位置、加速度等），reward设置等。
- environment⽂件是强化学习算法中的经典环境接口（step,reset, reward等）。你可以按照给定任务重写 environment.py ⽂件中的 self._set_action 函数。

## 要求

通过深度学习算法或者其他方法使被追击者能躲避追击者并到达目标点。深度强化学习算法训练可参考文件 `ppo.py`

要使用图像作为状态输入。

考核时会公布追击者和被追击者的初始位置，目标点的位置。

障碍物位置固定。

课程展示后将报告以小组形式发送到邮箱：`1075361077@qq.com`
