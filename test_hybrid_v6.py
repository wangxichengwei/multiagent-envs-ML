"""Behavioral checks for the v6 waypoint-offset controller."""
import unittest

import numpy as np

from hybrid_v3 import EvasionController
from hybrid_v6 import V6Controller, V6Config, PPOV6
from visual_state import VisualState


def perception(predator=(.6, .6), prey=(-.6, -.6), goal=(.6, -.6)):
    tracker = VisualState()
    obs = tracker.update(dict(prey=np.array(prey), predator=np.array(predator),
                              checkpoint=np.array(goal),
                              obstacles=[np.array(p) for p in ((-.35, .35), (.35, .35), (0., -.35))]))
    return tracker, obs


class V6Tests(unittest.TestCase):
    def test_zero_offset_matches_v3_rule(self):
        tracker, legacy = perception(predator=(0., -.6))
        v6 = V6Controller(tracker)
        v6.prepare(tracker, legacy)
        v3 = EvasionController(tracker)
        v3.prepare(tracker, legacy)
        np.testing.assert_array_equal(v6.execute(tracker, 0, np.zeros(2)),
                                      v3.execute(tracker, 0, np.zeros(2)))

    def test_offset_changes_direction_when_threatened(self):
        tracker, legacy = perception(predator=(0., -.6))
        v6 = V6Controller(tracker)
        v6.prepare(tracker, legacy)
        v6.threat = 1.0
        base = v6.execute(tracker, 0, np.zeros(2))
        moved = v6.execute(tracker, 0, np.array([1.0, 0.0]))
        self.assertGreater(np.linalg.norm(base - moved), 1e-3)

    def test_offset_gated_off_when_safe(self):
        tracker, legacy = perception(predator=(.7, .7), goal=(-.3, -.6))
        v6 = V6Controller(tracker)
        v6.prepare(tracker, legacy)
        v6.threat = 0.0
        base = v6.execute(tracker, 0, np.zeros(2))
        moved = v6.execute(tracker, 0, np.array([1.0, 0.0]))
        np.testing.assert_array_equal(base, moved)

    def test_config_offset_scale(self):
        cfg = V6Config()
        self.assertEqual(cfg.waypoint_offset_scale, 0.35)
        self.assertEqual(cfg.evasion_reward_weight, 4.0)

    def test_act_returns_finite_zero_command(self):
        import torch
        torch.set_num_threads(1)
        agent = PPOV6()
        command, _ = agent.act(np.zeros(76, dtype=np.float32), training=False)
        self.assertTrue(np.isfinite(command).all())
        np.testing.assert_allclose(command, np.zeros(2), atol=1e-6)


if __name__ == '__main__':
    unittest.main()
