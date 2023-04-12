import torch
import copy
import abc
import numpy as np
from typing import Union
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
from config import device


def transmission_cost_at_node(node, iter, full_size, transmitted_size):
    # This cost cannot be made random as of now, called both within and outside algorithm
    rs = RandomState(MT19937(SeedSequence(iter * 800 + node + 23456)))

    # Use node = -1 for server
    if node >= 0:
        const = 0.05
        if transmitted_size == 0:
            return 0.0
        else:
            return const + float(transmitted_size)/full_size / np.log2(1 + rs.chisquare(2))
    else:
        # smaller loss for the server
        const = 0.01
        if transmitted_size == 0:
            return 0.0
        else:
            return const + 0.2 * float(transmitted_size)/full_size / np.log2(1 + rs.chisquare(2))


class CompressedUpdate(abc.ABC):
    def __init__(self, node):
        self.node = node

    def get_transmitted_and_residual(self, iter, w_tmp: Union[torch.Tensor, None], w_residual_updates_at_node: torch.Tensor):
        """
        In place updates to transmitted vectors
        :param w_tmp:
        :param w_residual_updates_at_node:
        :return:
        """
        if w_tmp is not None:
            w_tmp += w_residual_updates_at_node
        else:
            w_tmp = w_residual_updates_at_node

        transmitted_indices, not_transmitted_indices = self._get_transmitted_indices(iter, w_tmp)

        w_tmp_residual = copy.deepcopy(w_tmp)
        w_tmp[not_transmitted_indices] = 0

        w_tmp_residual -= w_tmp

        return w_tmp, w_tmp_residual

    @abc.abstractmethod
    def _get_transmitted_indices(self, iter, vec: torch.Tensor):
        """
        Returns indices of transmitted and not transmitted elements from update vector
        :param vec:
        :return: transmitted_indices, not_transmitted_indices
        """
        raise NotImplementedError()


class CompressedLyapunov(CompressedUpdate):
    def __init__(self, node, target_average_cost, v, init_queue):
        super().__init__(node)
        self.target_average_cost = target_average_cost
        self.v = v
        self.queue = init_queue

    def _get_transmitted_indices(self, iter, vec: torch.Tensor):
        dim = vec.shape[0]

        vec_sq = torch.square(vec)
        vec_sq_sorted, sorted_indices = torch.sort(vec_sq, descending=True)

        penalty_no_transmit = self.v * torch.sum(vec_sq) - self.queue * self.target_average_cost  # Value for not transmitting anything (zero cost)

        # Assuming linear cost when at least one element is transmitted
        cost_delta = self.queue * (transmission_cost_at_node(self.node, iter, dim, 2) - transmission_cost_at_node(self.node, iter, dim, 1))

        tmp = torch.arange(vec_sq.shape[0], device=device)
        tmp2 = tmp[self.v * vec_sq_sorted <= cost_delta]
        if len(tmp2) > 0:
            i = tmp2[0]
        else:
            i = vec_sq.shape[0]
        drift_plus_penalty = self.v * torch.sum(vec_sq_sorted[i:]) + \
                             self.queue * (transmission_cost_at_node(self.node, iter, dim, i) - self.target_average_cost)
        if drift_plus_penalty < penalty_no_transmit:
            transmitted_elements = i
        else:
            transmitted_elements = 0

        self.queue += transmission_cost_at_node(self.node, iter, dim, transmitted_elements) - self.target_average_cost
        self.queue = max(0.001, self.queue)  # Do not allow negative queues, set to small value to avoid sudden blow up

        return sorted_indices[:transmitted_elements], sorted_indices[transmitted_elements:]


class CompressedNoneOrFixedRandom(CompressedUpdate):
    def __init__(self, node, target_average_cost, amount_of_transmission=1.0):
        super().__init__(node)
        self.target_average_cost = target_average_cost
        self.amount_of_transmission = amount_of_transmission

    def _get_transmitted_indices(self, iter, vec: torch.Tensor):
        dim = vec.shape[0]

        vec_sq = torch.square(vec)
        vec_sq_sorted, sorted_indices = torch.sort(vec_sq, descending=True)

        cost_no_transmit = transmission_cost_at_node(self.node, iter, dim, 0)  # Should be zero

        num_transmit = int(np.round(vec.shape[0] * self.amount_of_transmission))
        non_zero_items = torch.count_nonzero(vec_sq).item()
        if num_transmit > non_zero_items:
            num_transmit = non_zero_items

        cost_transmit = transmission_cost_at_node(self.node, iter, dim, num_transmit)

        if cost_no_transmit > 0:
            raise Exception('Cost when not transmitting should be zero')

        if cost_transmit > 0.0:
            threshold = self.target_average_cost / cost_transmit
        else:
            # Nothing to transmit
            threshold = 1.0

        threshold = min(1.0, threshold)

        if np.random.binomial(1, threshold) == 1:
            transmitted_elements = num_transmit
        else:
            transmitted_elements = 0

        return sorted_indices[:transmitted_elements], sorted_indices[transmitted_elements:]

