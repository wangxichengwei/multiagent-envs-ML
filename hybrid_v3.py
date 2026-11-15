"""Rule-based predator evasion on top of the v2 threat-aware controller.

v2 already predicts, over a 3-second pursuit preview, the minimum surface gap
for the keep route and for optional left/right side routes, but leaves the
keep/left/right choice to PPO, which empirically never switched. This module
replaces that learned choice with a deterministic safety rule:

  * in safe scenes (predicted threat at or below the gate) it is exactly the
    v2 zero-instruction base controller;
  * when the keep route is predicted to be intercepted, it automatically
    follows the side route whose preview predicts the largest surface gap,
    provided the gain is at least `gap_margin` and the route is obstacle-safe.

The preview uses only image-derived positions, visual velocity estimates and
the known dynamics; the rule never reads simulator ground truth and never
changes the physics or the base speed profile.
"""
from dataclasses import dataclass
import numpy as np

from hybrid_v2 import ThreatController, V2Config

VERSION = 'hybrid76-v3'


@dataclass(frozen=True)
class V3Config(V2Config):
    # Minimum predicted surface-gap gain required before switching from keep to
    # a side route, in world units. Keeps the rule from flapping on near-ties.
    gap_margin: float = 0.03


class EvasionController(ThreatController):
    """v2 threat controller plus a deterministic keep/left/right evasion rule."""

    def __init__(self, tracker, cfg=None):
        super().__init__(tracker, cfg or V3Config())
        self.rule_route = 0
        self.rule_telemetry = {}

    def _rule_decision(self, tracker):
        self.rule_route = 0
        self.rule_telemetry = {}
        if not self.decision:
            return
        paths = [self.current]
        route_of = [0]
        for side in (1, 2):
            if self.candidates[side] is not None:
                paths.append(self.candidates[side])
                route_of.append(side)
        if len(paths) == 1:
            return
        predictions = self.preview(tracker, paths)
        margin = getattr(self.cfg, 'gap_margin', 0.03)
        keep_gap = float(predictions[0, 0])
        side_gaps = {1: None, 2: None}
        best = None
        for j, side in enumerate(route_of):
            if side == 0:
                continue
            gap = float(predictions[j, 0])
            side_gaps[side] = gap
            # Candidate routes were already filtered by predicted obstacle
            # clearance; re-check here so the rule never picks a route whose
            # preview flags contact.
            if float(predictions[j, 4]) < 0.005:
                continue
            if gap > keep_gap + margin:
                if best is None or gap > best[0]:
                    best = (gap, side)
        if best is not None:
            self.rule_route = best[1]
        self.rule_telemetry = dict(keep_gap=keep_gap, side_gaps=side_gaps,
                                   chosen=self.rule_route, threat=float(self.threat))

    def prepare(self, tracker, legacy_state):
        obs = super().prepare(tracker, legacy_state)
        self._rule_decision(tracker)
        return obs

    def execute(self, tracker, route=0, command=None):
        if route == 0 and self.decision and self.rule_route:
            route = self.rule_route
        return super().execute(tracker, route, command)
