"""Image-derived visibility graph and potential-based training reward."""
import heapq
import numpy as np


class PathReward:
    def __init__(self, tracker, gamma=.99, beta=2., risk_weight=0.):
        self.gamma, self.beta, self.risk_weight = gamma, beta, risk_weight
        self.goal = tracker.positions['checkpoint'].copy()
        self.obstacles = tracker.obstacles.copy()
        self.radius = .09  # obstacle .05 + ball .025 + safety margin .015
        angles = np.arange(16) * (2*np.pi/16)
        ring = np.column_stack([np.cos(angles), np.sin(angles)])
        # Circumscribed polygon edges remain outside the inflated circle.
        ring *= self.radius / np.cos(np.pi/16) + 1e-4
        nodes = [self.goal]
        for obstacle in self.obstacles:
            nodes.extend(p for p in obstacle + ring if np.max(np.abs(p)) <= .825)
        self.nodes = np.array(nodes)
        self.costs = np.full(len(nodes), np.inf)
        self.costs[0] = 0
        edges = [[] for _ in nodes]
        for i in range(len(nodes)):
            for j in range(i):
                if self.visible(self.nodes[i], self.nodes[j]):
                    distance = np.linalg.norm(self.nodes[i]-self.nodes[j])
                    edges[i].append((j,distance)); edges[j].append((i,distance))
        queue = [(0.,0)]
        while queue:
            cost, i = heapq.heappop(queue)
            if cost > self.costs[i]:
                continue
            for j, length in edges[i]:
                if cost + length < self.costs[j]:
                    self.costs[j] = cost + length
                    heapq.heappush(queue,(cost+length,j))

    def visible(self, start, end):
        delta = end-start
        fraction = np.clip(((self.obstacles-start) @ delta) /
                           max(float(delta @ delta),1e-12),0,1)
        nearest = start + fraction[:,None]*delta
        return bool(np.all(np.linalg.norm(self.obstacles-nearest,axis=1) >= self.radius-1e-8))

    def distance(self, position):
        position = np.asarray(position)
        if np.max(np.abs(position)) > .825:
            return 4.
        if np.linalg.norm(position-self.goal) < .075:
            return 0.
        paths = [np.linalg.norm(position-node)+cost for node,cost in zip(self.nodes,self.costs)
                 if np.isfinite(cost) and self.visible(position,node)]
        # Bounded fallback inside inflated obstacles / disconnected map. No NaNs.
        return min(4., max(0., min(paths)-.075)) if paths else 4.

    def reward(self, previous, current, outcome=None):
        before = -self.distance(previous[:2])
        after = 0. if outcome is not None else -self.distance(current[:2])
        terminal = 30. if outcome == 'success' else (-30. if outcome is not None else 0.)
        progress = self.beta * (self.gamma*after-before)
        # Optional small predictive penalty, disabled for the initial ablation.
        risk = 0.
        if current[23] > 0 and np.all(current[26:28] > .5):
            risk = -self.risk_weight * max(0., .12-current[25]*2)/.12
        return terminal - .01 + progress + risk
