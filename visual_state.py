"""Stateful image-only observation encoder; no environment state access."""
from itertools import permutations
import numpy as np


STATE_DIM = 32
FEATURES = {
    'prey_position': slice(0, 2), 'prey_velocity': slice(2, 4),
    'predator_relative_position': slice(4, 6),
    'relative_velocity': slice(6, 8), 'goal_relative_position': slice(8, 10),
    'obstacles_relative_position': slice(10, 16),
    'boundary_clearance': slice(16, 20), 'previous_action': slice(20, 22),
    'risk': slice(22, 26), 'detected': slice(26, 32),
}


def applied_action(action):
    """Match environment's unit-norm acceleration direction before accel scaling."""
    action = np.asarray(action, dtype=float).reshape(2)
    return action / max(1.0, np.linalg.norm(action))


class VisualState:
    def __init__(self, dt=0.1, velocity_alpha=0.6, max_prediction_steps=3):
        self.dt = dt
        self.alpha = velocity_alpha
        self.max_prediction_steps = max_prediction_steps
        self.positions = {}
        self.measured = {}
        self.last_seen = {}
        self.velocities = {name: np.zeros(2) for name in ('prey', 'predator')}
        self.obstacles = None
        self.frame = -1

    @staticmethod
    def point(value):
        if value is None:
            return None
        value = np.asarray(value, dtype=float)
        return value.copy() if value.shape == (2,) and np.isfinite(value).all() else None

    def update(self, detections, previous_action=None):
        self.frame += 1
        valid = []
        for name in ('prey', 'predator', 'checkpoint'):
            point = self.point(detections.get(name))
            valid.append(float(point is not None))
            if point is not None:
                if name in self.velocities and name in self.measured:
                    elapsed = (self.frame - self.last_seen[name]) * self.dt
                    velocity = (point - self.measured[name]) / elapsed
                    limit = 0.25 if name == 'prey' else 0.28
                    velocity *= min(1.0, limit / max(np.linalg.norm(velocity), 1e-12))
                    self.velocities[name] = (self.alpha * velocity +
                                             (1 - self.alpha) * self.velocities[name])
                self.positions[name] = point
                self.measured[name] = point.copy()
                self.last_seen[name] = self.frame
            elif name in self.positions and name in self.velocities:
                if self.frame - self.last_seen[name] <= self.max_prediction_steps:
                    self.positions[name] += self.velocities[name] * self.dt
                else:
                    self.velocities[name] = np.zeros(2)

        points = [self.point(p) for p in detections.get('obstacles', [])]
        points = [p for p in points if p is not None][:3]
        obstacle_valid = np.zeros(3)
        if self.obstacles is None and len(points) == 3:
            # Geometric numbering, independent of connected-component area/order.
            self.obstacles = np.array(sorted(points, key=lambda p: (p[0], p[1])))
            obstacle_valid[:] = 1
        elif self.obstacles is not None and points:
            # One-to-one minimum-distance association, including partial detections.
            assignment = min(permutations(range(3), len(points)),
                key=lambda ids: sum(np.linalg.norm(p-self.obstacles[i])
                                    for p, i in zip(points, ids)))
            for point, index in zip(points, assignment):
                if np.linalg.norm(point-self.obstacles[index]) < 0.08:
                    obstacle_valid[index] = 1
            # Obstacles are static: keep initial image-derived centers to avoid
            # centroid drift when a moving ball partially occludes an obstacle.

        if len(self.positions) < 3 or self.obstacles is None:
            return None  # Initial map must be observed, never filled from world state.

        prey = self.positions['prey']
        relative = self.positions['predator'] - prey
        relative_velocity = self.velocities['predator'] - self.velocities['prey']
        distance = np.linalg.norm(relative)
        closing = -np.dot(relative, relative_velocity) / max(distance, 1e-12)
        speed_sq = np.dot(relative_velocity, relative_velocity)
        horizon = 1.0
        closest_time = (np.clip(-np.dot(relative, relative_velocity) / speed_sq, 0, horizon)
                        if speed_sq > 1e-10 else horizon)
        closest_distance = np.linalg.norm(relative + closest_time * relative_velocity)
        # Conservative clearances to the inner border for the ball's center.
        x, y = prey
        clearance = np.array([x + .825, .825 - x, y + .825, .825 - y]) / 2
        action = np.zeros(2) if previous_action is None else applied_action(previous_action)
        state = np.concatenate([
            prey, self.velocities['prey'] / .25, relative / 2,
            relative_velocity / .53, (self.positions['checkpoint'] - prey) / 2,
            ((self.obstacles - prey) / 2).reshape(-1), clearance, action,
            [max(0, distance-.05)/2, closing/.53, closest_time/horizon,
             max(0, closest_distance-.05)/2], valid, obstacle_valid,
        ])
        return np.clip(state, -1, 1).astype(np.float32)
