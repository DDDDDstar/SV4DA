import numpy as np


class Checker():
    def __init__(self, player_num, cache_size, threshold):
        self.player_num = player_num
        self.cache_size = cache_size
        self.threshold = threshold

    def convergence_check(self, SVs_var):
        if len(SVs_var) <= 0:
            return False
        if len(SVs_var[0]) < self.cache_size:
            return False

        count = 0
        sum_ = 0
        for (player_id, _) in SVs_var.items():
            latest_SV = SVs_var[player_id][-1]
            for ridx in range(2, self.cache_size+1):
                count += 1
                sum_ += np.abs((latest_SV-SVs_var[player_id][-ridx]) /
                               (latest_SV if latest_SV != 0 else 10**(-12)))

        convergence_diff = sum_/(count if count > 0 else 10**(-12))
        print("Current average convergence_diff (count %s): " % count,
              convergence_diff)
        if convergence_diff <= self.threshold:
            print("Convergence checking passed.")
            return True
        return False
