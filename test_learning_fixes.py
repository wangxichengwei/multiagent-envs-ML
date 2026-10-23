"""Headless regression tests: python -m unittest test_learning_fixes -v."""
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from agents.PPO.agent import PPO
from multiagent.environment import MultiAgentEnv
from multiagent.scenarios.simple_tag import Scenario
import train


class LearningFixTests(unittest.TestCase):
    def setUp(self):
        self.scenario = Scenario()
        self.env = MultiAgentEnv(
            self.scenario.make_world(), self.scenario.reset_world,
            self.scenario.reward, self.scenario.observation,
            info_callback=self.scenario.info, done_callback=self.scenario.is_done)
        self.env.reset(np.array([[-0.6, -0.6], [0.6, 0.6]]), [0.6, -0.6])

    def agent(self):
        return PPO(train.STATE_DIM, 2, SimpleNamespace(device='cpu', lr=1e-4,
                   gamma=0.99, eps_clip=0.2, K_epochs=1, train=True))

    def test_action_applies_immediately_and_reset_clears_it(self):
        before = self.env.world.agents[1].state.p_pos.copy()
        self.env.step(np.array([0., 1., 0., 0., 0.]))
        self.assertGreater(self.env.world.agents[1].state.p_pos[0], before[0])
        self.env.reset()
        for agent in self.env.world.agents:
            np.testing.assert_array_equal(agent.action.u, np.zeros(2))
            np.testing.assert_array_equal(agent.state.p_vel, np.zeros(2))

    def test_collision_precedes_goal_and_done_is_boolean(self):
        world = self.env.world
        world.check[0].state.p_pos = world.agents[1].state.p_pos.copy()
        self.assertEqual(self.scenario.outcome(world), 'success')
        world.agents[0].state.p_pos = world.agents[1].state.p_pos.copy()
        self.assertEqual(self.scenario.outcome(world), 'caught')
        self.assertTrue(all(self.scenario.is_done(a, world) for a in world.agents))
        world.agents[0].state.p_pos = np.array([-0.6, -0.6])
        world.landmarks[0].state.p_pos = world.agents[1].state.p_pos.copy()
        self.assertEqual(self.scenario.outcome(world), 'obstacle_collision')

    def test_returns_bootstrap_timeout_without_crossing_episode(self):
        agent = self.agent()
        agent.buffer.rewards = [1., 2., 100.]
        agent.buffer.is_terminals = [False, False, True]
        agent.buffer.is_truncated = [False, True, False]
        agent.buffer.next_values = [0., 10., 0.]
        np.testing.assert_allclose(agent.discounted_returns(),
                                   [1 + .99 * (2 + .99 * 10), 2 + .99 * 10, 100])

    def test_load_synchronizes_both_policies(self):
        agent = self.agent()
        with tempfile.TemporaryDirectory(dir='.') as folder:
            agent.save(folder + '/', 7)
            other = self.agent()
            other.load(folder + '/', 7)
            for expected, current, old in zip(agent.old_policy.parameters(),
                    other.policy.parameters(), other.old_policy.parameters()):
                torch.testing.assert_close(expected, current)
                torch.testing.assert_close(expected, old)

    def test_single_transition_update_is_finite(self):
        agent = self.agent()
        state = np.zeros(train.STATE_DIM, dtype=np.float32)
        agent.select_action(state)
        agent.record_result(1., False, True, state)
        agent.update()
        self.assertFalse(agent.buffer.rewards)
        self.assertTrue(all(torch.isfinite(p).all() for p in agent.policy.parameters()))

    def test_fixed_obstacles_and_sampling_clearance(self):
        cfg = SimpleNamespace(reset_bounds=(-.8, .8), reset_min_sep=.25,
              obstacles=np.array([[-.35, .35], [.35, .35], [0., -.35]]))
        rng = np.random.RandomState(15)
        for _ in range(30):
            train.reset_env(self.env, cfg, rng)
            np.testing.assert_array_equal(
                [o.state.p_pos for o in self.env.world.landmarks], cfg.obstacles)
            points = [a.state.p_pos for a in self.env.world.agents] + [self.env.world.check[0].state.p_pos]
            for i, point in enumerate(points):
                for other in points[i+1:] + list(cfg.obstacles):
                    self.assertGreaterEqual(np.linalg.norm(point-other), .25)

    def test_training_flushes_partial_rollout(self):
        agent = self.agent()
        with tempfile.TemporaryDirectory(dir='.') as folder:
            cfg = SimpleNamespace(env_name='test', algo='PPO', device='cpu', seed=15,
                start_ep=0, train_eps=1, train_steps=1, rollout_steps=2048,
                gamma=.99, path_beta=2., risk_weight=0.,
                model_path=folder+'/', reset_bounds=(-.8,.8), reset_min_sep=.25,
                obstacles=np.array([[-.35,.35],[.35,.35],[0.,-.35]]))
            with patch.object(train, 'get_state', return_value=np.zeros(train.STATE_DIM, dtype=np.float32)), patch.object(train, 'PathReward', return_value=SimpleNamespace(reward=lambda *args: 1.)):
                train.train(cfg, self.env, agent)
            self.assertFalse(agent.buffer.rewards)


if __name__ == '__main__':
    unittest.main()

