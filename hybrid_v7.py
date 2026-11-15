"""v7: obstacle-safe 2D waypoint offset for larger detours.

v6 let PPO offset the v3 waypoint freely, which pushed targets into obstacle
regions and produced obstacle collisions that the v3 rule never had. v7 keeps
the same action (a 2D offset) but projects it onto a safe set before use:

  * the offset target must lie inside the conservative boundary;
  * the straight segment prey -> target must clear every inflated obstacle
    (the same `RouteGraph.visible` test the base controller trusts);
  * if the raw offset is unsafe, its magnitude is shrunk by bisection until it
    is safe, falling back to the unmodified v3 waypoint (zero offset).

So zero offset is still the exact v3 rule, and no offset can cause a new
obstacle collision that the rule would not already have taken.
"""
import bisect
import math
import os
import random
from dataclasses import asdict, dataclass

import numpy as np
import torch

from hybrid_v3 import EvasionController
from hybrid_v4 import EvasionReward, PPOV4, V4Config, SCHEMA
from visual_state import applied_action

VERSION = 'hybrid76-v7'
ACTIONS = {'route': 'rule', 'continuous': ['waypoint_offset_x', 'waypoint_offset_y'],
           'transform': 'tanh', 'offset_scale': 0.35, 'safety': 'visible_projection'}


@dataclass(frozen=True)
class V7Config(V4Config):
    waypoint_offset_scale: float = 0.35
    evasion_reward_weight: float = 4.0
    offset_shrink_steps: int = 6


class V7Controller(EvasionController):
    """v3 rule + obstacle-safe learned waypoint offset."""

    def __init__(self, tracker, cfg=None):
        super().__init__(tracker, cfg or V7Config())

    def safe_offset(self, prey, waypoint, offset):
        """Return the largest safe multiple of `offset` on the segment [0, offset]."""
        scale = float(self.cfg.waypoint_offset_scale)

        def candidate(fraction):
            return np.clip(waypoint + offset * scale * fraction, -0.8, 0.8)

        def usable(target):
            inside = bool(np.max(np.abs(target)) <= 0.8)
            return inside and bool(self.graph.visible(prey, [target])[0])

        if not np.any(offset):
            return waypoint
        target = candidate(1.0)
        if usable(target):
            return target
        # Bisect between the known-safe fraction 0 and the unsafe fraction 1.
        low, high = 0.0, 1.0
        for _ in range(int(self.cfg.offset_shrink_steps)):
            middle = (low + high) / 2
            if usable(candidate(middle)):
                low = middle
            else:
                high = middle
        return candidate(low) if low > 0 else waypoint

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
        effective = command if self.threat > .05 else np.zeros(2)
        target = self.safe_offset(prey, self.current[0], effective)
        delta = target - prey
        distance = float(np.linalg.norm(delta))
        direction = delta / max(distance, 1e-9)
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


class PPOV7(PPOV4):
    def save(self, path, step, extra=None):
        if self.buffer:
            raise ValueError('Save only at completed rollout boundaries')
        state = dict(version=VERSION, features=SCHEMA, actions=ACTIONS, controller=asdict(V7Config()),
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
                or state.get('actions') != ACTIONS or state.get('controller') != asdict(V7Config())):
            raise ValueError('Incompatible v7 checkpoint')
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
