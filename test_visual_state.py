"""Visual state regression tests, including image resizing and missed detections."""
import unittest
import cv2
import numpy as np
from detect import detect_positions
from visual_state import VisualState, STATE_DIM, FEATURES


def scene(prey=(0., 0.), predator=(.4, 0.)):
    return dict(prey=np.array(prey), predator=np.array(predator),
                checkpoint=np.array([.6, -.6]),
                obstacles=[np.array(p) for p in [(-.35,.35),(.35,.35),(0.,-.35)]])


class VisualStateTests(unittest.TestCase):
    def test_velocity_and_closing_risk(self):
        tracker = VisualState(velocity_alpha=1)
        initial = tracker.update(scene())
        np.testing.assert_array_equal(initial[FEATURES['prey_velocity']], [0, 0])
        state = tracker.update(scene((.02, 0), (.38, 0)), [.8, .8])
        self.assertEqual(state.shape, (STATE_DIM,))
        self.assertTrue(np.isfinite(state).all())
        np.testing.assert_allclose(state[FEATURES['prey_velocity']], [.8, 0], atol=1e-6)
        np.testing.assert_allclose(state[FEATURES['relative_velocity']], [-.4/.53, 0], atol=1e-6)
        gap, closing, time, future_gap = state[FEATURES['risk']]
        self.assertGreater(closing, 0)
        self.assertLess(future_gap, gap)
        self.assertAlmostEqual(time, .9, places=5)
        np.testing.assert_allclose(state[FEATURES['previous_action']], [2**-.5]*2)

    def test_obstacle_order_and_partial_occlusion(self):
        tracker = VisualState()
        original = tracker.update(scene())
        res = scene()
        res['obstacles'].reverse()
        reordered = tracker.update(res)
        np.testing.assert_array_equal(original, reordered)
        res['obstacles'] = [res['obstacles'][0]]
        partial = tracker.update(res)
        np.testing.assert_array_equal(partial[10:16], original[10:16])
        self.assertEqual(partial[29:32].sum(), 1)
        res['obstacles'] = []
        missing = tracker.update(res)
        np.testing.assert_array_equal(missing[10:16], original[10:16])
        np.testing.assert_array_equal(missing[29:32], [0,0,0])

    def test_missing_dynamic_objects_and_reacquisition(self):
        tracker = VisualState(velocity_alpha=1)
        tracker.update(scene())
        tracker.update(scene((.01, 0), (.39, 0)))
        res = scene()
        res['prey'] = res['predator'] = res['checkpoint'] = None
        missing = tracker.update(res)
        np.testing.assert_array_equal(missing[26:29], [0,0,0])
        self.assertAlmostEqual(missing[0], .02, places=6)
        recovered = tracker.update(scene((.03, 0), (.37, 0)))
        self.assertAlmostEqual(recovered[2], .4, places=6)
        for _ in range(8):
            state = tracker.update(res)
        self.assertTrue(np.isfinite(state).all())
        np.testing.assert_array_equal(state[2:4], [0,0])
        # A fresh episode has no map, position or velocity from the previous one.
        self.assertIsNone(VisualState().update(res))

    def test_initialization_requires_complete_image_map(self):
        tracker = VisualState()
        res = scene()
        res['obstacles'] = res['obstacles'][:2]
        self.assertIsNone(tracker.update(res))
        self.assertIsNotNone(tracker.update(scene()))

    def test_approaching_and_receding_are_distinct(self):
        closing = VisualState(velocity_alpha=1)
        receding = VisualState(velocity_alpha=1)
        closing.update(scene()); receding.update(scene())
        approach = closing.update(scene(predator=(.38,0)))
        away = receding.update(scene(predator=(.42,0)))
        self.assertGreater(approach[23], 0)
        self.assertLess(away[23], 0)
        self.assertAlmostEqual(away[24], 0)

    def test_image_detection_at_both_resolutions(self):
        res = scene(prey=(-.6,-.6), predator=(.6,.6))
        image = np.full((800,800,3), 255, dtype=np.uint8)
        def pixel(p):
            return tuple(np.round([(p[0]+1)*400,(1-p[1])*400]).astype(int))
        for point in res['obstacles']:
            cv2.circle(image, pixel(point), 20, (64,64,64), -1)
        cv2.circle(image, pixel(res['prey']), 10, (172,236,172), -1)
        cv2.circle(image, pixel(res['predator']), 10, (236,172,172), -1)
        x,y = pixel(res['checkpoint'])
        cv2.rectangle(image, (x-10,y-10), (x+10,y+10), (204,153,204), -1)
        for size in (800,400):
            detected = detect_positions(cv2.resize(image,(size,size), interpolation=cv2.INTER_AREA), scale=size/2)
            for name in ('prey','predator','checkpoint'):
                np.testing.assert_allclose(detected[name], res[name], atol=.005)
            state = VisualState().update(detected)
            self.assertIsNotNone(state)
            np.testing.assert_array_equal(state[26:32], np.ones(6))


if __name__ == '__main__':
    unittest.main()
