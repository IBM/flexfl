import numpy as np
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence


def participation_cost_at_node(node, iter):
    rs = RandomState(MT19937(SeedSequence(iter * 1000 + node)))
    return rs.random()   # random cost between 0.0 and 1.0 when participating


class ParticipationLyapunov:
    def __init__(self, node, target_average_cost, v, init_queue):
        self.node = node
        self.target_average_cost = target_average_cost
        self.v = v
        self.queue = init_queue

    def get_participation(self, iter):
        current_cost = participation_cost_at_node(self.node, iter)
        if self.queue * current_cost <= 0.0:
            q_opt = 1.0
        else:
            q_opt = min(1.0, np.sqrt(self.v) / np.sqrt(self.queue * current_cost))
        self.queue += q_opt * current_cost - self.target_average_cost
        self.queue = max(0.0, self.queue)
        return q_opt


class ParticipationStatic:
    def __init__(self, node, target_average_cost):
        self.node = node
        self.target_average_cost = target_average_cost

    def get_participation(self, iter):
        mean_cost = 0.5  # Assumes this is known, see definition of participation_cost_at_node()
        q_opt = min(1.0, self.target_average_cost / mean_cost)
        return q_opt
