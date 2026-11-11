"""Monte Carlo tree search (UCT) with random rollouts.

Scores are min-max normalized over everything observed so far, so the
exploration constant does not depend on the scale of the objective.
"""

from __future__ import annotations

import math
import random
from typing import Generic, Optional

from .base import E, S, SearchStrategy


class _Node(Generic[S, E]):
    __slots__ = ("state", "parent", "children", "expanded", "visits", "total", "value")

    def __init__(self, state: S, parent: Optional["_Node[S, E]"], value: float) -> None:
        self.state = state
        self.parent = parent
        self.children: list[_Node[S, E]] = []
        self.expanded = False
        self.visits = 0
        self.total = 0.0
        self.value = value


class MctsSearch(SearchStrategy[S, E]):
    """UCT search: select by upper confidence bound, expand, roll out, back up.

    Parameters
    ----------
    n_iterations
        Number of select/expand/rollout/backup cycles.
    exploration
        UCT constant ``c`` in ``mean + c * sqrt(log N_parent / N_child)``.
    rollout_depth
        Maximum number of random moves per rollout (``None``: until a leaf).
    seed
        Seed of the generator used for rollouts and for picking the expanded child.
    """

    def __init__(
        self,
        n_iterations: int = 1000,
        exploration: float = 1.4,
        rollout_depth: Optional[int] = None,
        seed: Optional[int] = None,
        record_complete_only: bool = False,
    ) -> None:
        super().__init__(record_complete_only)
        if n_iterations < 1:
            raise ValueError("n_iterations must be >= 1")
        self.n_iterations = n_iterations
        self.exploration = exploration
        self.rollout_depth = rollout_depth
        self.rng = random.Random(seed)


    def _observe(self, value: float) -> None:
        self._lo = min(self._lo, value)
        self._hi = max(self._hi, value)

    def _normalized(self, value: float) -> float:
        if self._hi <= self._lo:
            return 0.5
        return (value - self._lo) / (self._hi - self._lo)

    def _uct(self, node: _Node[S, E], log_parent_visits: float) -> float:
        if node.visits == 0:
            return float("inf")
        exploit = self._normalized(node.total / node.visits)
        return exploit + self.exploration * math.sqrt(log_parent_visits / node.visits)



    def _expand_node(self, heuristic, node: _Node[S, E]) -> None:

        node.expanded = True
        elements = self._expand(heuristic, node.state)
        if not elements:
            return
        states = [heuristic.extend(node.state, e) for e in elements]
        for child_state, value in zip(states, self._score_many(heuristic, states)):
            child = _Node(child_state, node, value)
            self._observe(value)
            self._record(heuristic, child_state, value)
            node.children.append(child)

    def _rollout(self, heuristic, state: S) -> float:

        depth = 0
        limit = self.rollout_depth
        while not heuristic.is_complete(state):
            if limit is not None and depth >= limit:
                break
            elements = heuristic.candidates(state)
            if not elements:
                break
            state = heuristic.extend(state, self.rng.choice(list(elements)))
            depth += 1
        value = self._score(heuristic, state)
        self._observe(value)
        self._record(heuristic, state, value)
        return value

    @staticmethod
    def _backpropagate(node: Optional[_Node[S, E]], value: float) -> None:
        while node is not None:
            node.visits += 1
            node.total += value
            node = node.parent


    def search(self, initial, heuristic, *, budget=None):
        self._reset()
        self._lo, self._hi = float("inf"), float("-inf")

        root_value = self._score(heuristic, initial)
        self._observe(root_value)
        self._record(heuristic, initial, root_value)
        root: _Node[S, E] = _Node(initial, None, root_value)

        for _ in range(self.n_iterations):
            if not self._budget_left(budget):
                break
            node = root


            while node.expanded and node.children:
                log_n = math.log(max(node.visits, 1))
                node = max(node.children, key=lambda c: self._uct(c, log_n))


            if not node.expanded and not heuristic.is_complete(node.state):
                self._expand_node(heuristic, node)
                if node.children:
                    node = self.rng.choice(node.children)

            value = (node.value if heuristic.is_complete(node.state)
                     else self._rollout(heuristic, node.state))
            self._backpropagate(node, value)

        return self._result()
