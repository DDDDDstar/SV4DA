import random
import numpy as np


class Sampler():
    def __init__(self, sampling, algo, player_num):
        self.sampling = sampling
        self.algo = algo
        self.player_num = player_num
        self.sampling_times = 0
        self.permutations = set()
        self.coalitions = set()

    def generateRandomPermutation(self, permutation):
        while ",".join(map(str, permutation)) in self.permutations:
            random.shuffle(permutation)
        return permutation

    def MC_sample(self, last):
        if last is None:
            raise Exception("Parameter(last) is required.")
        if self.sampling == 'random':
            permutation = self.generateRandomPermutation(last)
        elif self.sampling == 'antithetic':
            permutation = self.generateRandomPermutation(last) if self.sampling_times % 2 == 1  \
                else list(reversed(last))
        elif self.sampling == 'stratified':
            permutation = self.generateRandomPermutation(last) if self.sampling_times % len(last) == 1  \
                else last[-1:] + last[:-1]
        self.permutations.add(",".join(map(str, permutation)))
        return permutation

    def MLE_sample(self, q, I_mq, m):
        if len(self.coalitions) >= 2**self.player_num:
            return [], True
        # generate Bernoulli random numbers independently
        if q is None or I_mq is None or m is None:
            raise Exception("Parameters(q, I_mq, and m) are required.")
        if self.sampling == 'antithetic' and m % 2 == 1:
            I_mq = 1 - I_mq
        else:
            I_mq = np.random.binomial(1, q, size=(self.player_num))
            while ",".join(map(str, I_mq)) in self.coalitions:
                q = np.random.rand()
                I_mq = np.random.binomial(1, q, size=(self.player_num))
        self.coalitions.add(",".join(map(str, I_mq)))
        return I_mq, False

    def generateRandomSubset(self, q_k):
        k = np.random.choice(range(1, self.player_num), p=q_k, size=1)
        selected_players = np.random.choice(
            range(self.player_num), int(k), replace=False)
        while ",".join(map(str, sorted(selected_players))) in self.coalitions:
            k = np.random.choice(range(0, self.player_num+1), size=1)
            selected_players = np.random.choice(
                range(self.player_num), int(k), replace=False)
        return selected_players

    def GT_sample(self, q_k, last):
        if self.sampling == 'antithetic':
            selected_players = self.generateRandomSubset(q_k)   \
                if self.sampling_times % 2 == 1 \
                else [player_id
                      for player_id in range(self.player_num)
                      if player_id not in last]
        else:
            selected_players = self.generateRandomSubset(q_k)
        self.coalitions.add(",".join(map(str, sorted(selected_players))))
        return selected_players, len(self.coalitions) >= 2**self.player_num

    def sample(self, **kwargs):
        self.sampling_times += 1
        if not callable(self.sampling):
            if self.algo in ['MC', 'CP', 'RE']:
                return self.MC_sampling(kwargs.get('last')), self.sampling_times
            elif self.algo == 'MLE':
                return self.MLE_sampling(kwargs.get('q'), kwargs.get('I_mq'), kwargs.get('m'))
            elif self.algo == 'GT':
                return self.GT_sampling(kwargs.get('q_k'), kwargs.get('last')), self.sampling_times
        else:
            return self.sampling()
