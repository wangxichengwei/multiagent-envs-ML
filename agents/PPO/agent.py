import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
# from model import ActorCritic
from agents.PPO.model import ActorCritic
class RolloutBuffer:
    def __init__(self) -> None:
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
        self.is_truncated = []
        self.next_values = []
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]
        del self.is_truncated[:]
        del self.next_values[:]
class PPO:
    def __init__(self, input_dim,action_dim,cfg):
        self.policy = ActorCritic(input_dim, action_dim).to(cfg.device)
        self.old_policy = ActorCritic(input_dim, action_dim).to(cfg.device)
        self.old_policy.load_state_dict(self.policy.state_dict())

        self.optimizer = optim.Adam(self.policy.parameters(), lr=cfg.lr)
        self.gamma = cfg.gamma
        self.eps_clip = cfg.eps_clip
        self.buffer = RolloutBuffer()
        self.device = cfg.device
        self.K_epochs = cfg.K_epochs
        self.train = cfg.train

        self.gae_lambda = getattr(cfg, 'gae_lambda', .95)
        self.batch_size = getattr(cfg, 'batch_size', 256)
        self.entropy_coef = getattr(cfg, 'entropy_coef', .01)
        self.max_grad_norm = getattr(cfg, 'max_grad_norm', .5)
        self.target_kl = getattr(cfg, 'target_kl', .03)
        self.last_metrics = {}
        self.MseLoss = nn.MSELoss()
   
    def select_action(self, state):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            distribution, value = self.old_policy.distribution(state)
            latent = distribution.sample() if self.train else distribution.mean
            action = self.old_policy.squash(latent)
            if self.train:
                # Store pre-transform samples for numerically stable likelihoods.
                self.buffer.actions.append(latent)
                self.buffer.states.append(state)
                self.buffer.logprobs.append(self.old_policy.log_prob(distribution, latent))
                self.buffer.state_values.append(value)
        return action

    def record_result(self, reward, terminated, truncated, next_state):
        self.buffer.rewards.append(float(reward))
        self.buffer.is_terminals.append(bool(terminated))
        self.buffer.is_truncated.append(bool(truncated))
        with torch.no_grad():
            state = torch.as_tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
            _, value = self.old_policy(state)
        self.buffer.next_values.append(float(value.item()))

    def discounted_returns(self):
        rewards = []
        discounted_reward = 0
        for i in reversed(range(len(self.buffer.rewards))):
            reward = self.buffer.rewards[i]
            is_terminal = bool(np.any(self.buffer.is_terminals[i]))
            if is_terminal:
                discounted_reward = 0
            elif self.buffer.is_truncated and self.buffer.is_truncated[i]:
                discounted_reward = self.buffer.next_values[i]
            elif i == len(self.buffer.rewards) - 1 and self.buffer.next_values:
                discounted_reward = self.buffer.next_values[i]
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        return rewards

    def compute_gae(self):
        values = torch.cat(self.buffer.state_values).reshape(-1).detach()
        advantages = torch.zeros_like(values)
        gae = 0.
        for i in reversed(range(len(self.buffer.rewards))):
            terminated = bool(np.any(self.buffer.is_terminals[i]))
            truncated = bool(self.buffer.is_truncated[i]) if self.buffer.is_truncated else False
            next_value = self.buffer.next_values[i] if self.buffer.next_values else (
                float(values[i+1]) if i+1 < len(values) else 0.)
            delta = self.buffer.rewards[i] + self.gamma * (not terminated) * next_value - values[i]
            gae = delta + self.gamma * self.gae_lambda * (not (terminated or truncated)) * gae
            advantages[i] = gae
        return advantages, advantages + values

    def update(self):
        if not self.buffer.rewards:
            return
        advantages, returns = self.compute_gae()
        advantages = (advantages-advantages.mean()) / (advantages.std(unbiased=False)+1e-8)
        states = torch.cat(self.buffer.states).detach()
        latent = torch.cat(self.buffer.actions).detach()
        old_logprobs = torch.cat(self.buffer.logprobs).detach()
        totals = []
        early_stop = False
        for _ in range(self.K_epochs):
            for ids in torch.randperm(len(states), device=self.device).split(self.batch_size):
                distribution, value = self.policy.distribution(states[ids])
                logprob = self.policy.log_prob(distribution, latent[ids])
                log_ratio = logprob-old_logprobs[ids]
                ratio = log_ratio.exp()
                kl = ((ratio-1)-log_ratio).mean()
                if self.target_kl and kl.item() > self.target_kl:
                    early_stop = True
                    break
                surrogate = torch.minimum(ratio*advantages[ids],
                    ratio.clamp(1-self.eps_clip, 1+self.eps_clip)*advantages[ids])
                value_loss = self.MseLoss(value.reshape(-1), returns[ids])
                # Reparameterized Monte Carlo entropy of the transformed policy.
                sample = distribution.rsample()
                entropy = -self.policy.log_prob(distribution, sample).mean()
                loss = -surrogate.mean() + .5*value_loss-self.entropy_coef*entropy
                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                with torch.no_grad():
                    self.policy.log_std.clamp_(np.log(.05),np.log(1.))
                totals.append([loss.item(), value_loss.item(), entropy.item(), kl.item(), float(grad_norm)])
            if early_stop:
                break
        if totals:
            self.last_metrics = dict(zip(('loss','value_loss','entropy','kl','grad_norm'),
                                         np.mean(totals,axis=0).tolist()))
        self.last_metrics['early_stop'] = early_stop
        self.last_metrics['samples'] = len(states)
        self.last_metrics['minibatches'] = len(totals)
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

    def save(self,path,i_ep):
        torch.save(self.old_policy.state_dict(), path + "PPO_model_"+str(i_ep))
        torch.save(self.optimizer.state_dict(), path + "PPO_optimizer_"+str(i_ep))

    def load(self, path,i_ep):
        weights = torch.load(path+ "PPO_model_"+str(i_ep), map_location=self.device, weights_only=True)
        self.old_policy.load_state_dict(weights)
        self.policy.load_state_dict(weights)
        if self.train:
            self.optimizer.load_state_dict(torch.load(path+ "PPO_optimizer_"+str(i_ep),
                                                      map_location=self.device, weights_only=True))
        self.buffer.clear()
