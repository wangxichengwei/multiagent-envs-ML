import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch

import train
from agents.PPO.agent import PPO
from agents.PPO.model import ActorCritic
from path_reward import PathReward


class PathPPOTests(unittest.TestCase):
    def agent(self):
        return PPO(32,2,SimpleNamespace(device='cpu',lr=1e-4,gamma=.99,
            eps_clip=.2,K_epochs=2,train=True,batch_size=2,target_kl=0))

    def planner(self):
        return PathReward(SimpleNamespace(positions={'checkpoint':np.array([.4,0.])},
            obstacles=np.array([[0.,0.],[.6,.6],[-.6,.6]])))

    def test_detour_rewards_progress_away_from_euclidean_goal(self):
        model = self.planner()
        before, after = np.zeros(32), np.zeros(32)
        before[:2] = [-.1,0]
        after[:2] = [-.1,.02]
        self.assertGreater(np.linalg.norm(after[:2]-model.goal),
                           np.linalg.norm(before[:2]-model.goal))
        self.assertLess(model.distance(after[:2]), model.distance(before[:2]))
        self.assertGreater(model.reward(before,after), model.reward(before,before))
        self.assertFalse(model.visible(before[:2],model.goal))

    def test_reward_terminal_priority_and_finite_fallback(self):
        model = self.planner()
        state = np.zeros(32)
        self.assertTrue(np.isfinite(model.reward(state,state)))
        self.assertGreater(model.reward(state,state,'success'), 0)
        self.assertLess(model.reward(state,state,'caught'), 0)
        self.assertLess(model.reward(state,state,'obstacle_collision'), 0)

    def test_transformed_density_matches_jacobian_and_norm(self):
        model = ActorCritic(32,2)
        latent = torch.tensor([.7,-1.2],requires_grad=True)
        jacobian = torch.autograd.functional.jacobian(model.squash,latent)
        expected_logdet = -2*torch.log1p(latent.square().sum())
        torch.testing.assert_close(torch.linalg.det(jacobian).log(),expected_logdet)
        dist,_ = model.distribution(torch.zeros(1,32))
        torch.testing.assert_close(model.log_prob(dist,latent),
            dist.log_prob(latent).sum(-1)-expected_logdet)
        samples = torch.randn(1000,2)*10
        self.assertTrue((model.squash(samples).norm(dim=-1) <= 1).all())

    def test_gae_masks_timeout_terminal_and_rollout_boundary(self):
        agent = self.agent()
        agent.buffer.state_values = [torch.tensor([[v]]) for v in (1.,2.,3.)]
        agent.buffer.rewards = [1.,2.,100.]
        agent.buffer.is_terminals = [False,False,True]
        agent.buffer.is_truncated = [False,True,False]
        agent.buffer.next_values = [2.,10.,999.]
        advantages,returns = agent.compute_gae()
        expected = [1+.99*2-1+.99*.95*(2+.99*10-2), 2+.99*10-2, 100-3]
        np.testing.assert_allclose(advantages.numpy(),expected,rtol=1e-6)
        np.testing.assert_allclose(returns.numpy(),np.array(expected)+[1,2,3],rtol=1e-6)
        agent.buffer.is_terminals[-1] = False
        agent.buffer.next_values[-1] = 5.
        self.assertAlmostEqual(agent.compute_gae()[0][-1].item(),100+.99*5-3,places=4)

    def test_minibatch_updates_and_learnable_exploration(self):
        agent = self.agent()
        initial = agent.policy.log_std.detach().clone()
        for i in range(5):
            state = np.full(32,i*.01,dtype=np.float32)
            action = agent.select_action(state)
            self.assertLessEqual(action.norm().item(),1.)
            agent.record_result(float(i),i==4,False,state)
        old_logprobs = torch.cat(agent.buffer.logprobs)
        dist,_ = agent.policy.distribution(torch.cat(agent.buffer.states))
        torch.testing.assert_close(agent.policy.log_prob(dist,torch.cat(agent.buffer.actions)),old_logprobs)
        agent.update()
        self.assertEqual(agent.last_metrics['minibatches'],6)
        self.assertFalse(torch.equal(initial,agent.policy.log_std))
        self.assertTrue(all(torch.isfinite(p).all() for p in agent.policy.parameters()))

    def test_training_updates_mid_episode_by_step_count(self):
        agent = self.agent()
        scenario = train.scenarios.load('simple_tag.py').Scenario()
        env = train.MultiAgentEnv(scenario.make_world(),scenario.reset_world,
            scenario.reward,scenario.observation,scenario.info,scenario.is_done)
        with tempfile.TemporaryDirectory(dir='.') as folder:
            cfg = SimpleNamespace(env_name='test',algo='PPO',device='cpu',seed=15,
                start_ep=0,train_eps=1,train_steps=5,rollout_steps=2,
                gamma=.99,path_beta=2.,risk_weight=0.,model_path=folder+'/',
                reset_bounds=(-.8,.8),reset_min_sep=.25,
                obstacles=np.array([[-.35,.35],[.35,.35],[0.,-.35]]))
            batches=[]
            update=agent.update
            def tracked_update():
                batches.append(len(agent.buffer.rewards)); update()
            with patch.object(agent,'update',side_effect=tracked_update), patch.object(
                train,'get_state',return_value=np.zeros(32,dtype=np.float32)), patch.object(
                train,'PathReward',return_value=SimpleNamespace(reward=lambda *args: 1.)):
                train.train(cfg,env,agent)
            self.assertEqual(batches,[2,2,1])


if __name__ == '__main__':
    unittest.main()
