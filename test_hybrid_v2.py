"""Behavioral regression checks for the independent threat-aware experiment."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from hybrid_control import HybridController
from hybrid_v2 import ThreatController, PPOV2, training_case
from hybrid_ppo import HybridPPO
from visual_state import VisualState


def perception(predator=(.6,.6),prey=(-.6,-.6),goal=(.6,-.6)):
    tracker=VisualState()
    obs=tracker.update(dict(prey=np.array(prey),predator=np.array(predator),checkpoint=np.array(goal),
        obstacles=[np.array(p) for p in ((-.35,.35),(.35,.35),(0.,-.35))]))
    return tracker,obs


class V2Tests(unittest.TestCase):
    def test_safe_scene_is_reference_and_commands_ignored(self):
        tracker,legacy=perception(predator=(.7,.7),goal=(-.3,-.6))
        control=ThreatController(tracker);obs=control.prepare(tracker,legacy)
        self.assertEqual(obs.shape,(76,));self.assertTrue(np.isfinite(obs).all())
        self.assertLessEqual(control.threat,.05)
        reference=HybridController(tracker);reference.prepare(tracker,legacy)
        np.testing.assert_array_equal(control.execute(tracker,0,[.7,-.8]),reference.execute(tracker))
        np.testing.assert_array_equal(obs[:32],legacy)
        np.testing.assert_array_equal(obs[68:],np.zeros(8))

    def test_threat_routes_predictions_and_decision_gate(self):
        tracker,legacy=perception(predator=(0.,-.6))
        control=ThreatController(tracker);obs=control.prepare(tracker,legacy)
        self.assertGreater(control.threat,.9);self.assertTrue(control.decision)
        self.assertGreater(control.mask.sum(),1)
        choice=int(np.flatnonzero(control.mask[1:])[0]+1)
        waypoint=control.candidates[choice][0].copy()
        control.execute(tracker,choice)
        self.assertEqual(control.switches,1)
        for _ in range(4):
            control.prepare(tracker,legacy)
            self.assertFalse(control.decision)
            np.testing.assert_array_equal(control.current[0],waypoint)
            with self.assertRaises(ValueError):control.execute(tracker,choice)
            control.execute(tracker)
        control.prepare(tracker,legacy)
        self.assertTrue(control.age>=5)

    def test_speed_recovery_accelerates_and_can_stop(self):
        for cap in (.0625,.125,.25):
            speeds=[ThreatController.commanded_speed(cap,c) for c in (-1,-.5,0,1)]
            self.assertEqual(speeds[0],0)
            self.assertTrue(all(a<b for a,b in zip(speeds[:3],speeds[1:3])))
            self.assertAlmostEqual(speeds[-1],cap)
        tracker,legacy=perception(predator=(0.,-.6))
        tracker.velocities['prey']=np.array([.23,0.])
        control=ThreatController(tracker);control.prepare(tracker,legacy)
        stop=control.execute(tracker,0,[0,-1])
        self.assertLess(stop[0],0)
        zero=control.execute(tracker,0,[0,0]);fast=control.execute(tracker,0,[0,1])
        self.assertGreater(fast[0],zero[0])

    def test_zero_commands_preserve_reference_even_under_threat(self):
        tracker,legacy=perception(predator=(0.,-.6))
        reference=HybridController(tracker);control=ThreatController(tracker)
        for _ in range(12):
            reference.prepare(tracker,legacy);control.prepare(tracker,legacy)
            np.testing.assert_array_equal(control.execute(tracker),reference.execute(tracker))

    def test_prediction_stops_at_goal_and_distinguishes_routes(self):
        tracker,legacy=perception(predator=(-.6,-.6),prey=(0.,-.6),goal=(.3,-.6))
        control=ThreatController(tracker)
        result=control.preview(tracker,[[np.array([.3,-.6])]])
        self.assertGreater(result[0,0],.25)
        tracker,legacy=perception(predator=(0.,-.6))
        control=ThreatController(tracker)
        control.prepare(tracker,legacy);pool,sides=control.proposal_pool(tracker)
        result=control.preview(tracker,pool)
        self.assertGreater(np.ptp(result[:,0]),.01)

    def test_sampler_curriculum_and_valid_physics(self):
        for step,fraction in ((0,.25),(16384,.5)):
            rng=np.random.RandomState(15);count=0
            for _ in range(160):
                case=training_case(rng,step);count+=case['sampling']=='intercept'
                points=np.array(case['agents']+[case['goal']]);obstacles=np.array(case['obstacles'])
                self.assertLessEqual(np.abs(points).max(),.8)
                self.assertGreaterEqual(np.linalg.norm(points[:,None]-obstacles,axis=2).min(),.25)
                self.assertGreaterEqual(min(np.linalg.norm(points[i]-points[j]) for i in range(3) for j in range(i)),.25)
            self.assertLess(abs(count/160-fraction),.12)

    def test_policy_probabilities_update_and_checkpoint(self):
        torch.set_num_threads(1);torch.manual_seed(15)
        tracker,legacy=perception(predator=(0.,-.6));control=ThreatController(tracker)
        obs=control.prepare(tracker,legacy);agent=PPOV2()
        routes,_,_=agent.policy.distributions(torch.tensor(obs).unsqueeze(0))
        self.assertLess(float(routes.probs[0,0].detach()),.8)
        for _ in range(64):
            route,cmd,sample=agent.act(obs,True)
            lp,_,_=agent.old_policy.evaluate(torch.tensor(obs).unsqueeze(0),torch.tensor([route]),torch.tensor(sample['latent']).unsqueeze(0),False)
            self.assertAlmostEqual(float(lp.detach()),sample['logprob'],places=5)
            agent.record(sample,float(route!=0),True,False,obs,[0,0])
        agent.update();self.assertTrue(np.isfinite(agent.last_metrics['loss']))
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            path=Path(folder)/'v2.pt';agent.save(path,64)
            other=PPOV2();other.load(path,True)
            self.assertEqual(agent.act(obs)[0],other.act(obs)[0])
            np.testing.assert_array_equal(agent.act(obs)[1],other.act(obs)[1])
            path=Path(folder)/'v1.pt';HybridPPO().save(path,0)
            with self.assertRaises(ValueError):other.load(path)
        safe=obs.copy();safe[64]=.4;safe[53:56]=[1,0,0]
        lp,_,ent=agent.policy.evaluate(torch.tensor(safe).unsqueeze(0),torch.tensor([0]),torch.tensor([[.2,-.3]]))
        self.assertEqual(float(lp.detach()),0);self.assertEqual(float(ent.detach()),0)


if __name__=='__main__':unittest.main()
