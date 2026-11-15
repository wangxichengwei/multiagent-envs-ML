import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from hybrid_control import HybridController, ControllerConfig, risk_metrics
from hybrid_reward import SoftPathReward
from hybrid_ppo import HybridPPO, HybridActorCritic
from visual_state import VisualState, applied_action

def perception(prey=(-.6,-.6),goal=(.6,-.6)):
    tracker=VisualState()
    obs=tracker.update(dict(prey=np.array(prey),predator=np.array([.6,.6]),
        checkpoint=np.array(goal),obstacles=[np.array(x) for x in ((-.35,.35),(.35,.35),(0.,-.35))]))
    return tracker,obs

class HybridTests(unittest.TestCase):
    def test_zero_command_is_velocity_baseline(self):
        tracker,obs=perception();control=HybridController(tracker)
        state=control.prepare(tracker,obs)
        action=control.execute(tracker)
        np.testing.assert_array_equal(state[:32],obs)
        self.assertEqual(state.shape,(64,));self.assertTrue(np.isfinite(state).all())
        np.testing.assert_allclose(action,[1.,0.])
        agent=HybridPPO();route,cmd,_=agent.act(state)
        self.assertEqual(route,0);np.testing.assert_array_equal(cmd,[0,0])
        control=HybridController(tracker);control.prepare(tracker,obs)
        np.testing.assert_array_equal(control.execute(tracker,route,cmd),action)

    def test_turn_speed_and_reverse_braking(self):
        tracker,_=perception();control=HybridController(tracker)
        _,straight=control.profile(np.array([0.,0.]),[np.array([.04,0.]),np.array([.4,0.])])
        angle,corner=control.profile(np.array([0.,0.]),[np.array([.04,0.]),np.array([.04,.4])])
        self.assertAlmostEqual(angle,np.pi/2);self.assertLess(corner,straight)
        tracker.velocities['prey']=np.array([.25,0.]);tracker.positions['checkpoint']=np.array([-.7,-.6])
        obs=np.zeros(32);control=HybridController(tracker);control.prepare(tracker,obs)
        self.assertLess(control.execute(tracker)[0],0)

    def test_route_gates_masks_and_recovery(self):
        tracker,obs=perception(prey=(-.3,-.35),goal=(.4,-.35))
        control=HybridController(tracker);state=control.prepare(tracker,obs)
        self.assertTrue(control.decision)
        choice=int(np.flatnonzero(control.mask[1:])[0]+1)
        control.execute(tracker,choice)
        self.assertEqual(control.switches,1)
        for _ in range(4):
            state=control.prepare(tracker,obs)
            self.assertFalse(control.decision)
            with self.assertRaises(ValueError): control.execute(tracker,choice)
            control.execute(tracker,0)
        control.prepare(tracker,obs);self.assertTrue(control.age>=5)
        tracker.positions['prey']=np.array([0.,-.35])
        state=control.prepare(tracker,obs)
        self.assertTrue(np.isfinite(state).all());self.assertTrue(control.mask[0])
        self.assertFalse(control.decision)
        self.assertGreater(np.linalg.norm(control.execute(tracker)),0)

    def test_explicit_detour_persists_with_clear_goal(self):
        tracker,obs=perception()
        control=HybridController(tracker);control.prepare(tracker,obs)
        self.assertTrue(control.decision)
        choice=int(np.flatnonzero(control.mask[1:])[0]+1)
        waypoint=control.candidates[choice][0].copy()
        control.execute(tracker,choice)
        control.prepare(tracker,obs)
        np.testing.assert_array_equal(control.current[0],waypoint)
        self.assertFalse(control.decision)

    def test_obstacle_metrics_and_ttc(self):
        metrics=risk_metrics([.2,0],[-.25,0],.075)
        np.testing.assert_allclose(metrics,[.125,.25,0,.5],atol=1e-7)
        self.assertEqual(risk_metrics([.2,0],[.25,0],.075)[3],1)

    def test_reward_continuity_and_no_route_bonus(self):
        tracker,_=perception();reward=SoftPathReward(tracker)
        for radius in (.075,.09,.10):
            a=np.array([radius-1e-5,-.35]);b=np.array([radius+1e-5,-.35])
            self.assertLess(abs(reward.distance(a)-reward.distance(b)),.01)
        inside=[reward.distance([x,-.35]) for x in np.linspace(0,.15,20)]
        self.assertTrue(np.isfinite(inside).all());self.assertGreater(np.ptp(inside),.01)
        a=np.zeros(64);a[:2]=[-.5,-.5];b=a.copy();b[59:62]=[0,1,0]
        self.assertEqual(reward.reward(a,a),reward.reward(a,b))
        self.assertLess(reward.reward(a,a,'caught')['total'],0)
        self.assertGreater(reward.reward(a,a,'success')['total'],0)

    def test_joint_logprob_masks_and_tanh_jacobian(self):
        model=HybridActorCritic();obs=torch.zeros(2,64)
        obs[:,53]=1;obs[1,54]=1
        route=torch.tensor([0,1]);latent=torch.tensor([[.2,-.3],[.2,-.3]])
        lp,_,_=model.evaluate(obs,route,latent,False)
        routes,normal,_=model.distributions(obs)
        corrected=normal.log_prob(latent).sum(-1)-torch.log(1-torch.tanh(latent).square()).sum(-1)
        torch.testing.assert_close(lp[0],corrected[0])
        torch.testing.assert_close(lp[1],corrected[1]+routes.log_prob(route)[1])
        obs[:,54]=0
        for _ in range(30): self.assertTrue((model.distributions(obs)[0].sample()==0).all())

    def test_gae_boundaries_update_and_checkpoint(self):
        torch.set_num_threads(1)
        agent=HybridPPO();tracker,legacy=perception();control=HybridController(tracker)
        obs=control.prepare(tracker,legacy)
        for i in range(3):
            _,_,sample=agent.act(obs,True)
            agent.record(sample,1.,i==2,i==1,obs,np.zeros(2))
        agent.buffer[0]['value']=1.;agent.buffer[0]['next_value']=2.
        agent.buffer[1]['value']=2.;agent.buffer[1]['next_value']=10.
        agent.buffer[2]['value']=3.;agent.buffer[2]['next_value']=999.
        advantages,returns=agent.gae()
        np.testing.assert_allclose(advantages,[1+.99*2-1+.99*.95*(1+.99*10-2),1+.99*10-2,-2],rtol=1e-6)
        agent.update();self.assertFalse(agent.buffer)
        self.assertTrue(all(torch.isfinite(p).all() for p in agent.policy.parameters()))
        with tempfile.TemporaryDirectory(dir='.') as directory:
            path=Path(directory)/'state.pt';agent.save(path,2048)
            expected=agent.act(obs)[1];expected_random=torch.rand(3)
            other=HybridPPO();state=other.load(path,resume=True)
            np.testing.assert_array_equal(other.act(obs)[1],expected)
            torch.testing.assert_close(torch.rand(3),expected_random)
            self.assertEqual(state['step'],2048)
            torch.save({'policy':{}},path)
            with self.assertRaises(ValueError): other.load(path)

if __name__=='__main__':unittest.main()
