# -*- coding: utf-8 -*-
"""探索脚本：验证渲染颜色与 像素->世界坐标 映射。"""
import numpy as np
import cv2

import multiagent.scenarios as scenarios
from multiagent.environment import MultiAgentEnv

W, H = 800, 800  # 渲染尺寸
SCALE = 400.0    # 每世界单位对应像素数（[-1,1] -> [0,800]）


def world_to_pixel(pos):
    """世界坐标[-1,1] -> 图像像素(row, col)。row=0 为图像顶部(y=+1)。"""
    x, y = pos
    col = (x + 1.0) * SCALE
    row = (1.0 - y) * SCALE
    return int(round(row)), int(round(col))


def pixel_to_world(row, col):
    x = col / SCALE - 1.0
    y = 1.0 - row / SCALE
    return np.array([x, y])


def main():
    scenario = scenarios.load("simple_tag.py").Scenario()
    world = scenario.make_world()
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward,
                        scenario.observation, info_callback=None,
                        done_callback=scenario.is_done, shared_viewer=True)

    obs_pos = [[-0.35, 0.35], [0.35, 0.35], [0, -0.35]]
    env.reset(np.array([[0.0, 0.0], [0.7, 0.7]]), [-0.5, -0.5], obs_pos)

    image = env.render("rgb_array")[0]  # (800,800,3) RGB uint8
    cv2.imwrite("render_sample.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    print("saved render_sample.png, image shape", image.shape, image.dtype)

    # 地面真值（仅用于验证）
    gt = {
        "predator": world.agents[0].state.p_pos,
        "prey":     world.agents[1].state.p_pos,
        "check":    world.check[0].state.p_pos,
        "obstacles": [lm.state.p_pos for lm in world.landmarks],
    }
    for name, p in gt.items():
        p = np.asarray(p, dtype=float)
        print(f"GT {name:10s} pos={p}")

    def sample(name, p):
        r, c = world_to_pixel(np.asarray(p, dtype=float))
        patch = image[max(0, r-1):r+2, max(0, c-1):c+2].reshape(-1, 3)
        print(f"{name:10s} pixel=({r},{c}) rgb(median)={np.median(patch, axis=0).astype(int)}")
        return r, c

    print("\n--- 各实体中心像素的颜色采样 ---")
    for name, p in gt.items():
        if name == "obstacles":
            for i, o in enumerate(p):
                sample(f"obstacle[{i}]", o)
        else:
            sample(name, p)

    # 采样几个 border 的颜色
    for i in (0, 20, 40, 60):
        sample(f"border[{i}]", world.borders[i].state.p_pos)

    # 背景颜色
    print(f"background corner rgb={image[5, 5]}  center rgb={image[400, 400]}")

    # 验证坐标映射：prey 在 [0.7,0.7]
    r, c = world_to_pixel([0.7, 0.7])
    print(f"\n坐标映射自检: world(0.7,0.7)->pixel({r},{c}); pixel({r},{c})->world {pixel_to_world(r,c)}")


if __name__ == "__main__":
    main()
