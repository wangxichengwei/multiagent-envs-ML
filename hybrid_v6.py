"""v6: PPO learns a 2D waypoint offset for larger detours.

v3 already decides *when* to detour and picks a left/right waypoint, but those
waypoints are capped at ~0.25 world units laterally. The remaining failures are
head-on interceptions that need a bigger dodge. v6 keeps the v3 route decision
and lets PPO learn a bounded 2D offset applied to the selected waypoint, so the
effective detour can be pushed further sideways or forwards. Zero offset is the
exact v3 rule.
"""
import math
import os
import random
from dataclasses import asdict, dataclass

import numpy as np
import torch

from hybrid_v3 import EvasionController
from hybrid_v4 import EvasionReward, PPOV4, V4Config, SCHEMA
from visual_state import applied_action

VERSION = 'hybrid76-v6'
ACTIONS = {'route': 'rule', 'continuous': ['waypoint_offset_x', 'waypoint_offset_y'],
           'transform': 'tanh', 'offset_scale': 0.35}


@dataclass(frozen=True)
class V6Config(V4Config):
    waypoint_offset_scale: float = 0.35   # max offset per axis, world units
    evasion_reward_weight: float = 4.0


class V6Controller(EvasionController):
    """v3 rule + learned 2D waypoint offset for larger detours."""

    def __init__(self, tracker, cfg=None):
        super().__init__(tracker, cfg or V6Config())

    def execute(self, tracker, route=0, command=None):
        command = np.zeros(2) if command is None else np.clip(np.asarray(command), -1, 1)
        if route == 0 and self.decision and self.rule_route:
            route = self.rule_route
        if route and (not self.decision or not self.mask[route]):
            raise ValueError('Invalid route')
        if self.decision:
            if route:
                self.current = self.candidates[route]
                self.manual_route = True
                self.side = route
                self.switches += 1
            self.age = 0
        prey = tracker.positions['prey']
        waypoint = self.current[0]
        # Gate the offset exactly like v4/v5: only when threatened.
        effective = command if self.threat > .05 else np.zeros(2)
        target = np.clip(waypoint + effective * self.cfg.waypoint_offset_scale, -0.8, 0.8)
        delta = target - prey
        distance = float(np.linalg.norm(delta))
        direction = delta / max(distance, 1e-9)
        # Corner-aware speed limit uses the original path turn and the offset distance.
        turn = 0.
        if len(self.current) > 1:
            next_delta = self.current[1] - self.current[0]
            turn = float(np.arccos(np.clip(
                np.dot(delta, next_delta) / max(distance * np.linalg.norm(next_delta), 1e-9), -1, 1)))
            corner = .25 * max(.25, math.cos(turn / 2) ** 2)
            cap = min(self.cfg.max_speed, math.sqrt(corner ** 2 + 2 * self.cfg.brake_accel * max(distance - self.cfg.reached, 0.)))
        else:
            cap = min(self.cfg.max_speed, math.sqrt(2 * self.cfg.brake_accel * max(distance - self.cfg.reached, 0.)))
        error = cap * direction - tracker.velocities['prey']
        action = applied_action(self.cfg.gain * error)
        self.previous_command = command.copy()
        self.age += 1
        return action


class PPOV6(PPOV4):
    def save(self, path, step, extra=None):
        if self.buffer:
            raise ValueError('Save only at completed rollout boundaries')
        state = dict(version=VERSION, features=SCHEMA, actions=ACTIONS, controller=asdict(V6Config()),
                     step=step, policy=self.policy.state_dict(), optimizer=self.optimizer.state_dict(),
                     torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(),
                     python_rng=random.getstate(), extra=extra or {})
        if torch.cuda.is_available():
            state['cuda_rng'] = torch.cuda.get_rng_state_all()
        temporary = str(path) + '.tmp'
        torch.save(state, temporary)
        os.replace(temporary, path)

    def load(self, path, resume=False):
        state = torch.load(path, map_location=self.device, weights_only=False)
        if (state.get('version') != VERSION or state.get('features') != SCHEMA
                or state.get('actions') != ACTIONS or state.get('controller') != asdict(V6Config())):
            raise ValueError('Incompatible v6 checkpoint')
        self.policy.load_state_dict(state['policy'])
        self.old_policy.load_state_dict(state['policy'])
        self.buffer.clear()
        if resume:
            self.optimizer.load_state_dict(state['optimizer'])
            torch.set_rng_state(state['torch_rng'].cpu())
            np.random.set_state(state['numpy_rng'])
            random.setstate(state['python_rng'])
            if 'cuda_rng' in state and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(state['cuda_rng'])
        return state
