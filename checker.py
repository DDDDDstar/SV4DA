import copy
import numpy as np


class Checker():
    def __init__(self, player_num, cache_size, threshold):
        self.player_num = player_num
        self.cache_size = cache_size
        self.threshold = threshold

        self.SV_cache = []

    def cache_SV(self, SVs):
        self.SV_cache.append(copy.deepcopy(SVs))
        return len(self.SV_cache) < self.cache_size

    def convergence_check(self):
        if self.cache_SV(SVs):
            return False

        count = 0
        sum_ = 0
        for SVs in self.SV_cache[-self.cache_size:-1]:
            for (player_id, SV) in SVs.items():
                count += 1
                latest_SV = self.SV_cache[-1][player_id]
                sum_ += np.abs((latest_SV-SV) /
                               (latest_SV if latest_SV != 0 else 10**(-12)))
        convergence_diff = sum_/(count if count > 0 else 10**(-12))
        print("Current average convergence_diff (count %s): " % count,
              convergence_diff)
        if convergence_diff <= self.threshold:
            print("Convergence checking passed.")
            return True
        return False
