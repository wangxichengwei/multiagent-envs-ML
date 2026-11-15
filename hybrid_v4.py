"""PPO on top of the v3 rule-based evasion controller (v4).

v3 already selects a safe route deterministically; v4 lets PPO learn only the
two continuous commands (heading offset, speed recovery) that are layered on
that route every step. The policy mean is initialised to zero, so at the first
step v4 is exactly the v3 rule (the strongest baseline we have), and PPO can
only make incremental, verifiable improvements.

The command is gated exactly like v2: when the keep route is predicted safe the
controller ignores the command, so the policy gradient and entropy for those
steps are masked out.
"""
import math
import os
import random
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F

from hybrid_v2 import SCHEMA, VERSION as V2_VERSION
from hybrid_v3 import EvasionController, V3Config
from hybrid_reward import SoftPathReward

VERSION = 'hybrid76-v4'
ACTIONS = {'route': 'rule', 'continuous': ['heading_offset_pi/2', 'braking_or_speed_recovery'],
           'transform': 'tanh', 'command_gate': 'threat>0.05'}


@dataclass(frozen=True)
class V4Config(V3Config):
    # Reward weight on the "increase predator clearance while threatened" term.
    evasion_reward_weight: float = 2.0


class EvasionReward(SoftPathReward):
    """v2 soft-distance potential plus a dense, threat-gated clearance reward.

    The extra term rewards increasing the predicted closest surface gap when the
    predator is already close, giving PPO an immediate signal for lateral dodges
    that the pure path potential cannot provide.
    """

    def __init__(self, tracker, gamma=.99, beta=2., evasion_reward_weight=2.0):
        super().__init__(tracker, gamma, beta)
        self.evasion_reward_weight = evasion_reward_weight

    def reward(self, previous, current, outcome=None, raw_reward=0.):
        parts = super().reward(previous, current, outcome, raw_reward)
        evasion = 0.
        # Feature 25 is the predicted closest surface gap divided by two.
        if np.all(current[26:28] > .5) and np.all(previous[26:28] > .5):
            prev_gap = max(0., float(previous[25] * 2))
            cur_gap = max(0., float(current[25] * 2))
            threat = float(np.clip(1 - prev_gap / .16, 0, 1))
            gain = float(np.clip(cur_gap - prev_gap, -.2, .2))
            evasion = self.evasion_reward_weight * threat * gain
        parts['predator_evasion'] = evasion
        parts['total'] = sum(value for key, value in parts.items() if key not in ('raw', 'total'))
        return parts


def network(output):
    return nn.Sequential(nn.Linear(76, 256), nn.ELU(), nn.Linear(256, 64), nn.ELU(),
                         nn.Linear(64, 32), nn.ELU(), nn.Linear(32, output))


class ActorCriticV4(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = network(2)
        self.critic = network(1)
        # Zero mean -> tanh(0)=0 -> exact v3 rule behaviour at initialisation.
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)
        self.log_std = nn.Parameter(torch.full((2,), math.log(.15)))

    def distributions(self, obs):
        continuous = Normal(self.actor(obs), self.log_std.clamp(math.log(.05), math.log(.5)).exp())
        return continuous, self.critic(obs).reshape(-1)

    @staticmethod
    def command_active(obs):
        return obs[:, 64] * 2 < .16 * .95

    def corrected_logprob(self, distribution, latent):
        logdet = 2 * (math.log(2) - latent - F.softplus(-2 * latent))
        return (distribution.log_prob(latent) - logdet).sum(-1)

    def evaluate(self, obs, latent, entropy=True):
        continuous, value = self.distributions(obs)
        active = self.command_active(obs)
        logprob = active * self.corrected_logprob(continuous, latent)
        ent = None
        if entropy:
            ent = active * (-self.corrected_logprob(continuous, continuous.rsample()))
        return logprob, value, ent


class PPOV4:
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        self.policy = ActorCriticV4().to(self.device)
        self.old_policy = ActorCriticV4().to(self.device)
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-4)
        self.buffer = []
        self.last_metrics = {}

    def act(self, obs, training=False):
        tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            continuous, value = self.old_policy.distributions(tensor)
            latent = continuous.sample() if training else continuous.mean
            logprob, _, _ = self.old_policy.evaluate(tensor, latent, entropy=False)
        sample = dict(obs=np.array(obs, copy=True), latent=latent[0].cpu().numpy(),
                      logprob=float(logprob.item()), value=float(value.item()),
                      command_active=bool(self.old_policy.command_active(tensor).item()))
        return torch.tanh(latent)[0].cpu().numpy(), sample

    def record(self, sample, reward, terminated, truncated, next_obs, executed):
        tensor = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            next_value = float(self.old_policy.critic(tensor).item())
        self.buffer.append(dict(sample, reward=float(reward), terminated=bool(terminated),
                                truncated=bool(truncated), next_value=next_value,
                                executed=np.array(executed, copy=True)))

    def gae(self):
        advantages = np.zeros(len(self.buffer), dtype=np.float32)
        last = 0.
        for i in reversed(range(len(self.buffer))):
            row = self.buffer[i]
            delta = row['reward'] + .99 * (not row['terminated']) * row['next_value'] - row['value']
            last = delta + .99 * .95 * (not (row['terminated'] or row['truncated'])) * last
            advantages[i] = last
        values = np.array([r['value'] for r in self.buffer], dtype=np.float32)
        return advantages, advantages + values

    def update(self):
        if not self.buffer:
            return
        advantages, returns = self.gae()
        to_tensor = lambda value, dtype=torch.float32: torch.as_tensor(np.asarray(value), dtype=dtype, device=self.device)
        obs = to_tensor([r['obs'] for r in self.buffer])
        latent = to_tensor([r['latent'] for r in self.buffer])
        old_log = to_tensor([r['logprob'] for r in self.buffer])
        adv = to_tensor(advantages)
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        targets = to_tensor(returns)
        metrics = []
        stop = False
        for _ in range(8):
            for ids in torch.randperm(len(obs), device=self.device).split(256):
                lp, value, entropy = self.policy.evaluate(obs[ids], latent[ids])
                logratio = lp - old_log[ids]
                ratio = logratio.exp()
                kl = ((ratio - 1) - logratio).mean()
                if kl.item() > .03:
                    stop = True
                    break
                policy_loss = -torch.minimum(ratio * adv[ids], ratio.clamp(.8, 1.2) * adv[ids]).mean()
                value_loss = F.mse_loss(value, targets[ids])
                loss = policy_loss + .5 * value_loss - .01 * entropy.mean()
                self.optimizer.zero_grad()
                loss.backward()
                grad = nn.utils.clip_grad_norm_(self.policy.parameters(), .5)
                self.optimizer.step()
                with torch.no_grad():
                    self.policy.log_std.clamp_(math.log(.05), math.log(.5))
                metrics.append([loss.item(), value_loss.item(), entropy.mean().item(), kl.item(), float(grad)])
            if stop:
                break
        self.last_metrics = dict(samples=len(obs), minibatches=len(metrics), early_stop=stop)
        if metrics:
            self.last_metrics.update(zip(('loss', 'value_loss', 'entropy', 'kl', 'grad_norm'),
                                         np.mean(metrics, axis=0).tolist()))
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

    def save(self, path, step, extra=None):
        if self.buffer:
            raise ValueError('Save only at completed rollout boundaries')
        state = dict(version=VERSION, features=SCHEMA, actions=ACTIONS, controller=asdict(V4Config()),
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
                or state.get('actions') != ACTIONS or state.get('controller') != asdict(V4Config())):
            raise ValueError('Incompatible v4 checkpoint')
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
