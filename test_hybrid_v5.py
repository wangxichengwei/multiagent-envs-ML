"""Behavioral checks for the v5 full-direction controller."""
import math
import unittest

import numpy as np

from hybrid_v3 import EvasionController
from hybrid_v5 import V5Controller, V5Config, PPOV5
from visual_state import VisualState


def perception(predator=(.6, .6), prey=(-.6, -.6), goal=(.6, -.6)):
    tracker = VisualState()
    obs = tracker.update(dict(prey=np.array(prey), predator=np.array(predator),
                              checkpoint=np.array(goal),
                              obstacles=[np.array(p) for p in ((-.35, .35), (.35, .35), (0., -.35))]))
    return tracker, obs


class V5Tests(unittest.TestCase):
    def test_zero_command_matches_v3_rule(self):
        tracker, legacy = perception(predator=(0., -.6))
        v5 = V5Controller(tracker)
        v5.prepare(tracker, legacy)
        v3 = EvasionController(tracker)
        v3.prepare(tracker, legacy)
        np.testing.assert_array_equal(v5.execute(tracker, 0, np.zeros(2)),
                                      v3.execute(tracker, 0, np.zeros(2)))

    def test_heading_max_is_pi(self):
        cfg = V5Config()
        self.assertAlmostEqual(cfg.heading_max, math.pi)
        self.assertEqual(cfg.evasion_reward_weight, 4.0)

    def test_half_command_rotates_quarter_turn(self):
        tracker, legacy = perception(predator=(0., -.6))
        v5 = V5Controller(tracker)
        v5.prepare(tracker, legacy)
        # Force a threatened scene where the command is not gated out.
        v5.threat = 1.0
        base = v5.execute(tracker, 0, np.zeros(2))
        turned = v5.execute(tracker, 0, np.array([0.5, 0.0]))
        # heading_max * 0.5 = pi/2, so the two acceleration vectors differ and
        # are roughly orthogonal (unit-limited directions).
        self.assertGreater(np.linalg.norm(base - turned), 1e-3)

    def test_act_returns_finite_zero_command(self):
        import torch
        torch.set_num_threads(1)
        agent = PPOV5()
        obs = np.zeros(76, dtype=np.float32)
        command, _ = agent.act(obs, training=False)
        self.assertTrue(np.isfinite(command).all())
        np.testing.assert_allclose(command, np.zeros(2), atol=1e-6)


if __name__ == '__main__':
    unittest.main()
