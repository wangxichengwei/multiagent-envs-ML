"""Behavioral checks for the v4 PPO-on-rule controller."""
import unittest

import numpy as np
import torch

from hybrid_v4 import EvasionReward, PPOV4, ActorCriticV4
from visual_state import VisualState


def perception(predator=(.6, .6), prey=(-.6, -.6), goal=(.6, -.6)):
    tracker = VisualState()
    obs = tracker.update(dict(prey=np.array(prey), predator=np.array(predator),
                              checkpoint=np.array(goal),
                              obstacles=[np.array(p) for p in ((-.35, .35), (.35, .35), (0., -.35))]))
    return tracker, obs


def make_obs(gap, closing=0., detected=True):
    obs = np.zeros(76, dtype=np.float32)
    obs[25] = gap / 2
    obs[23] = closing / .53
    if detected:
        obs[26] = 1.
        obs[27] = 1.
    return obs


class V4Tests(unittest.TestCase):
    def test_zero_mean_init_is_exact_rule(self):
        agent = PPOV4()
        command, sample = agent.act(np.zeros(76, dtype=np.float32), training=False)
        np.testing.assert_allclose(command, np.zeros(2), atol=1e-6)

    def test_command_active_gating(self):
        obs = torch.zeros(1, 76)
        obs[0, 64] = .05   # gap 0.10 -> threatened
        self.assertTrue(bool(ActorCriticV4.command_active(obs).item()))
        obs[0, 64] = .5    # gap 1.00 -> safe
        self.assertFalse(bool(ActorCriticV4.command_active(obs).item()))

    def test_evasion_reward_rewards_clearance_gain_under_threat(self):
        tracker, _ = perception()
        reward = EvasionReward(tracker, evasion_reward_weight=2.0)
        prev = make_obs(0.05)
        cur = make_obs(0.15)
        parts = reward.reward(prev, cur, None, 0.)
        # threat = 1 - .05/.16 = .6875; gain = .10; weight 2 -> .1375
        self.assertAlmostEqual(parts['predator_evasion'], 2.0 * (1 - .05 / .16) * .10, places=5)
        self.assertAlmostEqual(parts['total'], sum(v for k, v in parts.items() if k not in ('raw', 'total')), places=5)

    def test_evasion_reward_zero_when_safe_or_shrinking(self):
        tracker, _ = perception()
        reward = EvasionReward(tracker, evasion_reward_weight=2.0)
        safe = reward.reward(make_obs(0.5), make_obs(0.5), None, 0.)
        self.assertEqual(safe['predator_evasion'], 0.0)
        shrink = reward.reward(make_obs(0.15), make_obs(0.05), None, 0.)
        self.assertLessEqual(shrink['predator_evasion'], 0.0)

    def test_update_produces_finite_metrics(self):
        torch.set_num_threads(1)
        torch.manual_seed(15)
        agent = PPOV4()
        obs = np.zeros(76, dtype=np.float32)
        obs[64] = .05  # threatened, so commands are active
        for _ in range(64):
            command, sample = agent.act(obs, True)
            agent.record(sample, float(command[0] != 0), True, False, obs, [0, 0])
        agent.update()
        self.assertTrue(np.isfinite(agent.last_metrics['loss']))


if __name__ == '__main__':
    unittest.main()
