# -*- coding: utf-8 -*-
"""任务一：从渲染图像中提取各物体位置。

输入：env.render("rgb_array")[0] 返回的 (800, 800, 3) RGB uint8 图像。
输出：dict，包含 prey / predator / checkpoint / obstacles 的世界坐标。

坐标约定：
  - 世界坐标为 [-1, 1] x [-1, 1]，屏幕中心为原点 (0, 0)。
  - 图像左上角为 (row=0, col=0)，对应世界 (x=-1, y=+1)。
  - 转换：x = col/400 - 1,  y = 1 - row/400。

颜色依据（已在 explore_render.py 中实测确认）：
  - prey      绿色   RGB(172, 236, 172)
  - predator  粉色   RGB(236, 172, 172)   (亮，r>220)
  - checkpoint 紫色  RGB(204, 153, 204)
  - obstacle  深灰   RGB(64, 64, 64)
  - border    鲑红   RGB(204, 102, 102)   (r<=220，比 predator 暗)
"""
import numpy as np
import cv2

SCALE = 400.0  # 每个世界单位对应的像素数
MIN_AREA = 20  # 连通域最小面积（像素），用于滤除噪声


def pixel_to_world(row, col, scale=SCALE):
    """图像像素 (row, col) -> 世界坐标 [x, y]。scale 为每个世界单位对应的像素数。"""
    x = col / scale - 1.0
    y = 1.0 - row / scale
    return np.array([x, y], dtype=float)


def _blob_centroids(mask, topk=None, min_area=MIN_AREA):
    """返回 mask 中面积最大的 topk 个连通域的中心像素坐标 [(col, row), ...]。"""
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    if n <= 1:  # 只有背景
        return []
    blobs = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        # centroids[i] 为 (col, row) 即 (x, y)
        blobs.append((area, centroids[i]))
    blobs.sort(key=lambda t: -t[0])  # 按面积降序
    if topk is not None:
        blobs = blobs[:topk]
    return [c for _, c in blobs]


def detect_positions(image, scale=SCALE):
    """从 RGB 图像中检测各物体世界坐标。

    参数:
        image: (H, W, 3) RGB uint8 图像。
        scale: 每个世界单位对应的像素数（800x800 时为 400）。
              训练时可将图像缩小后传入，例如 400x400 -> scale=200。

    返回 dict:
        {
          'prey':       np.array([x, y]) 或 None,
          'predator':   np.array([x, y]) 或 None,
          'checkpoint': np.array([x, y]) 或 None,
          'obstacles':  [np.array([x, y]), ...]  (最多 3 个),
        }
    """
    r = image[:, :, 0].astype(np.int16)
    g = image[:, :, 1].astype(np.int16)
    b = image[:, :, 2].astype(np.int16)

    # 1. 猎物（绿色占优）
    mask_prey = (g - r > 30) & (g - b > 30)
    # 2. 障碍物（深灰：三通道都很暗，背景为纯白 255）
    mx = np.maximum(np.maximum(r, g), b)
    mask_obs = mx < 90
    # 3. 打卡点（紫色：蓝≈红，且蓝明显高于绿）
    mask_check = (np.abs(b - r) < 30) & (b - g > 25)
    # 4. 追击者（亮粉：红占优且很亮）
    mask_pred = (r - g > 40) & (r - b > 40) & (r > 220)
    # 5. 边界（鲑红：红占优但比追击者暗）
    mask_border = (r - g > 40) & (r - b > 40) & (r <= 220)

    def centroid_world(mask):
        cs = _blob_centroids(mask, topk=1)
        return pixel_to_world(cs[0][1], cs[0][0], scale) if cs else None

    result = {
        'prey': centroid_world(mask_prey),
        'predator': centroid_world(mask_pred),
        'checkpoint': centroid_world(mask_check),
        'obstacles': [pixel_to_world(c[1], c[0], scale)
                      for c in _blob_centroids(mask_obs, topk=3)],
    }
    return result


if __name__ == '__main__':
    import multiagent.scenarios as scenarios
    from multiagent.environment import MultiAgentEnv

    scenario = scenarios.load('simple_tag.py').Scenario()
    world = scenario.make_world()
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward,
                        scenario.observation, info_callback=None,
                        done_callback=scenario.is_done, shared_viewer=True)

    obs_pos = [[-0.35, 0.35], [0.35, 0.35], [0, -0.35]]
    env.reset(np.array([[0.0, 0.0], [0.7, 0.7]]), [-0.5, -0.5], obs_pos)
    image = env.render('rgb_array')[0]

    res = detect_positions(image)
    for k, v in res.items():
        print(f'{k:10s} {v}')
