import math
import numpy as np
import random

from abc import ABC, abstractmethod
from collections import defaultdict

from src.utils import RegistryMixin


class SearchStrategy(ABC, RegistryMixin):
    """
    Performs search to reduce a set of masks to a subset of K masks.
    """

    @abstractmethod
    def run_search(self, scores, k: int = 3) -> list:
        pass


class MCTS:
    "Monte Carlo tree searcher. First rollout the tree then choose a move."
    """
    ====================================
    ORIGINAL SOURCE
    ====================================
    A minimal implementation of Monte Carlo tree search (MCTS) in Python 3
    Luke Harold Miles, July 2019, Public Domain Dedication
    See also https://en.wikipedia.org/wiki/Monte_Carlo_tree_search
    https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1
    """

    def __init__(self, exploration_weight=1):
        self.Q = defaultdict(int)  # total reward of each node
        self.N = defaultdict(int)  # total visit count for each node
        self.children = dict()  # children of each node
        self.exploration_weight = exploration_weight

    def choose(self, node):
        "Choose the best successor of node. (Choose a move in the game)"
        if node.is_terminal():
            raise RuntimeError(f"choose called on terminal node {node}")

        if node not in self.children:
            return node.find_random_child()

        def score(n):
            if self.N[n] == 0:
                return float("-inf")  # avoid unseen moves
            return self.Q[n] / self.N[n]  # average reward

        return max(self.children[node], key=score)

    def do_rollout(self, node):
        "Make the tree one layer better. (Train for one iteration.)"
        path = self._select(node)
        leaf = path[-1]
        self._expand(leaf)
        reward = self._simulate(leaf)
        self._backpropagate(path, reward)

    def _select(self, node):
        "Find an unexplored descendent of `node`"
        path = []
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:
                # node is either unexplored or terminal
                return path
            unexplored = self.children[node] - self.children.keys()
            if unexplored:
                n = unexplored.pop()
                path.append(n)
                return path
            node = self._uct_select(node)  # descend a layer deeper

    def _expand(self, node):
        "Update the `children` dict with the children of `node`"
        if node in self.children:
            return  # already expanded
        self.children[node] = node.find_children()

    def _simulate(self, node):
        "Returns the reward for a random simulation (to completion) of `node`"
        while True:
            if node.is_terminal():
                reward = node.reward()
                return reward
            node = node.find_random_child()

    def _backpropagate(self, path, reward):
        "Send the reward back up to the ancestors of the leaf"
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward

            # Commented out as single-player game
            #reward = 1 - reward  # 1 for me is 0 for my enemy, and vice versa

    def _uct_select(self, node):
        "Select a child of node, balancing exploration & exploitation"

        # All children of node should already be expanded:
        assert all(n in self.children for n in self.children[node])

        log_N_vertex = math.log(self.N[node])

        def uct(n):
            "Upper confidence bound for trees"
            return self.Q[n] / self.N[n] + self.exploration_weight * math.sqrt(
                log_N_vertex / self.N[n]
            )

        return max(self.children[node], key=uct)



class State:
    
    def __init__(self, scores, k: int = 3, chosen_idxs: list = []):

        self.scores = scores
        self.k = k
        self.chosen_idxs = chosen_idxs

    def find_children(self):
        if self.terminal:
            return set()

        return {
            State(self.scores, k=self.k, chosen_idxs=self.chosen_idxs + [i]) for i in range(len(self.scores)) if i not in self.chosen_idxs
        }
        
    def find_random_child(self):
        if self.terminal:
            return None

        possible_idxs = [i for i in range(len(self.scores)) if i not in self.chosen_idxs]
        
        return State(self.scores, k=self.k, chosen_idxs=self.chosen_idxs + [random.choice(possible_idxs)])

    @property
    def terminal(self):
        return len(self.chosen_idxs) == self.k

    def reward(self):
        if not self.terminal:
            raise RuntimeError(f"reward called on nonterminal node {self}")

        D = self.scores[self.chosen_idxs][:, self.chosen_idxs]
        for i in range(len(D)):
            D[i, i] = 1.0

        return np.mean(np.max(self.scores[self.chosen_idxs], axis=-1)) - np.mean(np.max(D, axis=0), axis=-1).item()
         

    def __hash__(self):
        return hash(tuple(self.chosen_idxs))

    def __eq__(node1, node2):
        return node1.chosen_idxs == node2.chosen_idxs

    def is_terminal(self):
        return self.terminal


@SearchStrategy.register_subclass("mcts")
class MCTSSearch(SearchStrategy):

    def __init__(self, num_rollouts: int = 300) -> None:
        super().__init__()
        self.num_rollouts = num_rollouts

    def run_search(self, scores, k=3):
        root = State(scores, k=k)
        mcts = MCTS()
        for _ in range(k):
            for _ in range(self.num_rollouts):
                mcts.do_rollout(root)
            chosen_child = mcts.choose(root)
            root = chosen_child
        return root.chosen_idxs
