"""Behavioral checks for the obstacle-safe waypoint-offset controller."""
import unittest

import numpy as np

from hybrid_v3 import EvasionController
from hybrid_v7 import V7Controller, V7Config, PPOV7
from visual_state import VisualState


def perception(predator=(.6, .6), prey=(-.6, -.6), goal=(.6, -.6)):
    tracker = VisualState()
    obs = tracker.update(dict(prey=np.array(prey), predator=np.array(predator),
                              checkpoint=np.array(goal),
                              obstacles=[np.array(p) for p in ((-.35, .35), (.35, .35), (0., -.35))]))
    return tracker, obs


class V7Tests(unittest.TestCase):
    def test_zero_offset_matches_v3_rule(self):
        tracker, legacy = perception(predator=(0., -.6))
        v7 = V7Controller(tracker)
        v7.prepare(tracker, legacy)
        v3 = EvasionController(tracker)
        v3.prepare(tracker, legacy)
        np.testing.assert_array_equal(v7.execute(tracker, 0, np.zeros(2)),
                                      v3.execute(tracker, 0, np.zeros(2)))

    def test_unsafe_offset_is_shrunk_or_rejected(self):
        tracker, legacy = perception(predator=(0., -.6))
        v7 = V7Controller(tracker)
        v7.prepare(tracker, legacy)
        prey = tracker.positions['prey']
        waypoint = v7.current[0].copy()
        # Push the target straight at the nearest obstacle at (-0.35, 0.35).
        bad = np.array([-0.9, 0.9])
        target = v7.safe_offset(prey, waypoint, bad)
        # Whatever survives must still be visible from prey (no obstacle crossing).
        self.assertTrue(bool(v7.graph.visible(prey, [target])[0]))
        self.assertLessEqual(np.max(np.abs(target)), .8 + 1e-9)

    def test_safe_offset_keeps_full_magnitude(self):
        tracker, legacy = perception(predator=(0., -.6))
        v7 = V7Controller(tracker)
        v7.prepare(tracker, legacy)
        prey = tracker.positions['prey']
        waypoint = v7.current[0].copy()
        target = v7.safe_offset(prey, waypoint, np.zeros(2))
        np.testing.assert_array_equal(target, waypoint)

    def test_config(self):
        cfg = V7Config()
        self.assertEqual(cfg.waypoint_offset_scale, 0.35)
        self.assertEqual(cfg.evasion_reward_weight, 4.0)

    def test_act_returns_finite_zero_command(self):
        import torch
        torch.set_num_threads(1)
        agent = PPOV7()
        command, _ = agent.act(np.zeros(76, dtype=np.float32), training=False)
        self.assertTrue(np.isfinite(command).all())
        np.testing.assert_allclose(command, np.zeros(2), atol=1e-6)


if __name__ == '__main__':
    unittest.main()
