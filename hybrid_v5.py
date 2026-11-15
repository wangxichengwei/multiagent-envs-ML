"""v5: wider heading authority and stronger evasion reward for PPO.

The v4 continuous command was limited to +-90 deg, too small to dodge a predator
already sitting near the direct goal line. v5 lets the policy rotate the heading
up to +-180 deg (any direction) and doubles the threat-gated clearance reward,
so PPO can learn large lateral or even rearward dodges before re-approaching the
goal. The zero-mean init still reproduces the v3 rule exactly.
"""
import math
import os
import random
from dataclasses import asdict, dataclass

import numpy as np
import torch

from hybrid_v3 import EvasionController
from hybrid_v4 import EvasionReward, PPOV4, V4Config, SCHEMA

VERSION = 'hybrid76-v5'
ACTIONS = {'route': 'rule', 'continuous': ['heading_offset_pi', 'braking_or_speed_recovery'],
           'transform': 'tanh', 'command_gate': 'threat>0.05'}


@dataclass(frozen=True)
class V5Config(V4Config):
    heading_max: float = math.pi          # +-180 deg instead of +-90 deg
    evasion_reward_weight: float = 4.0


class V5Controller(EvasionController):
    """v3 rule + full-direction continuous heading offset."""

    def __init__(self, tracker, cfg=None):
        super().__init__(tracker, cfg or V5Config())

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
        _, cap = self.profile(tracker.positions['prey'], self.current)
        effective = command if self.threat > .05 else np.zeros(2)
        speed = self.commanded_speed(cap, float(effective[1]))
        direction = self.current[0] - tracker.positions['prey']
        direction = direction / max(np.linalg.norm(direction), 1e-9)
        angle = float(effective[0]) * self.cfg.heading_max
        c, s = math.cos(angle), math.sin(angle)
        direction = np.array([c * direction[0] - s * direction[1],
                              s * direction[0] + c * direction[1]])
        error = speed * direction - tracker.velocities['prey']
        along = max(0., float(error @ direction))
        boost = self.cfg.acceleration_boost * max(0., float(effective[1])) * along * direction
        from visual_state import applied_action
        action = applied_action(self.cfg.gain * (error + boost))
        self.previous_command = command.copy()
        self.age += 1
        return action


class PPOV5(PPOV4):
    def save(self, path, step, extra=None):
        if self.buffer:
            raise ValueError('Save only at completed rollout boundaries')
        state = dict(version=VERSION, features=SCHEMA, actions=ACTIONS, controller=asdict(V5Config()),
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
                or state.get('actions') != ACTIONS or state.get('controller') != asdict(V5Config())):
            raise ValueError('Incompatible v5 checkpoint')
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
