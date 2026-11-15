import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import evaluate_hybrid
import train_hybrid

class LiveEvaluationTests(unittest.TestCase):
    def test_normal_flip_and_scoped_restore(self):
        window=SimpleNamespace(has_exit=False,switch_to=Mock(),dispatch_events=Mock(),
            set_vsync=Mock(),set_caption=Mock(),flip=Mock())
        env=SimpleNamespace(viewers=[SimpleNamespace(window=window)],reset=Mock())
        original_state=train_hybrid.image_state;original_reset=env.reset
        def render_state(*args):
            window.flip()
            return 'image-derived-state'
        with patch.object(train_hybrid.train,'get_state',side_effect=render_state):
            with evaluate_hybrid.visible_evaluation(env,episode_pause=0):
                env.reset()
                self.assertEqual(train_hybrid.image_state(env,object()),'image-derived-state')
                window.flip.assert_called_once()
                window.switch_to.assert_called_once()
            self.assertIs(train_hybrid.image_state,original_state)
            self.assertIs(env.reset,original_reset)

    def test_error_restores_batch_function(self):
        env=SimpleNamespace(viewers=[],reset=Mock())
        original=train_hybrid.image_state
        with self.assertRaises(RuntimeError):
            with evaluate_hybrid.visible_evaluation(env):
                raise RuntimeError('test')
        self.assertIs(train_hybrid.image_state,original)

if __name__=='__main__': unittest.main()
