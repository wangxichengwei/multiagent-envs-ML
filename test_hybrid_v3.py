"""Behavioral checks for the rule-based evasion controller."""
import unittest

import numpy as np

from hybrid_v2 import ThreatController
from hybrid_v3 import EvasionController
from visual_state import VisualState


def perception(predator=(.6, .6), prey=(-.6, -.6), goal=(.6, -.6)):
    tracker = VisualState()
    obs = tracker.update(dict(prey=np.array(prey), predator=np.array(predator),
                              checkpoint=np.array(goal),
                              obstacles=[np.array(p) for p in ((-.35, .35), (.35, .35), (0., -.35))]))
    return tracker, obs


class V3Tests(unittest.TestCase):
    def test_safe_scene_matches_base_and_does_not_switch(self):
        tracker, legacy = perception(predator=(.7, .7), goal=(-.3, -.6))
        control = EvasionController(tracker)
        control.prepare(tracker, legacy)
        self.assertEqual(control.rule_route, 0)
        base = ThreatController(tracker)
        base.prepare(tracker, legacy)
        np.testing.assert_array_equal(control.execute(tracker), base.execute(tracker))

    def test_threat_scene_switches_to_a_safer_side(self):
        tracker, legacy = perception(predator=(0., -.6))
        control = EvasionController(tracker)
        control.prepare(tracker, legacy)
        self.assertTrue(control.decision)
        self.assertIn(control.rule_route, (1, 2))
        self.assertEqual(control.rule_telemetry['chosen'], control.rule_route)
        keep = control.rule_telemetry['keep_gap']
        side = control.rule_telemetry['side_gaps'][control.rule_route]
        self.assertGreater(side, keep + control.cfg.gap_margin)

    def test_manual_route_retained_after_rule_switch(self):
        tracker, legacy = perception(predator=(0., -.6))
        control = EvasionController(tracker)
        control.prepare(tracker, legacy)
        self.assertTrue(control.decision)
        waypoint = control.candidates[control.rule_route][0].copy()
        control.execute(tracker)  # applies the rule's side-route switch
        self.assertEqual(control.switches, 1)
        for _ in range(4):
            control.prepare(tracker, legacy)
            self.assertFalse(control.decision)
            np.testing.assert_array_equal(control.current[0], waypoint)


if __name__ == '__main__':
    unittest.main()
