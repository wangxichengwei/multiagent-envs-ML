"""Apply the visible-render contract to the independent v2 entry point."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import evaluate_hybrid_v2 as entry
import train_hybrid_v2 as experiment


class V2LiveTests(unittest.TestCase):
    def test_frames_flip_and_patch_is_scoped(self):
        window=SimpleNamespace(has_exit=False,switch_to=Mock(),dispatch_events=Mock(),
            set_vsync=Mock(),set_caption=Mock(),flip=Mock())
        env=SimpleNamespace(viewers=[SimpleNamespace(window=window)],reset=Mock())
        original_state=experiment.image_state;original_reset=env.reset
        def render(*args):
            window.flip()
            return 'visual-state'
        with patch.object(experiment.train,'get_state',side_effect=render):
            with entry.visible_evaluation(env,episode_pause=0):
                env.reset()
                self.assertEqual(experiment.image_state(env,object()),'visual-state')
                window.flip.assert_called_once();window.switch_to.assert_called_once()
        self.assertIs(experiment.image_state,original_state);self.assertIs(env.reset,original_reset)

    def test_failure_restores_batch_helper(self):
        env=SimpleNamespace(viewers=[],reset=Mock());original=experiment.image_state
        with self.assertRaises(RuntimeError):
            with entry.visible_evaluation(env):raise RuntimeError('expected')
        self.assertIs(experiment.image_state,original)


if __name__=='__main__':unittest.main()
