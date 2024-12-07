import copy
import math
import time
import numpy as np


def quantization(SVs, level):
    # level=0~1
    level = int(len(SVs)*level)
    sorted_SV = sorted(SVs.values())
    interval = int(np.ceil(len(SVs)/level))
    quantization_map = dict()
    for level_index in range(level):
        index_lower = level_index*interval
        index_higher = min(len(SVs), (level_index+1)*interval)
        for sv_idx in range(index_lower, index_higher):
            quantization_map[
                sorted_SV[sv_idx]
            ] = sorted_SV[int((index_lower + index_higher)/2)]
        if index_higher >= len(SVs):
            break
    SVs = dict([(k, quantization_map[v]) for (k, v) in SVs.items()])
    return SVs


def differential_privacy(SVs, level):
    noise = np.random.normal(0, level, len(SVs))
    SVs = dict([(k, v + noise[idx])
                for idx, (k, v) in enumerate(SVs.items())])
    return SVs


def dimension_reduction(SVs, level, reference_var=dict()):
    if len(reference_var) == len(SVs):
        exposed_idx = dict(
            sorted(reference_var.items(),
                   key=lambda item: item[1])[-int(len(reference_var)*level):]
        ).keys()

    else:
        exposed_idx = dict(
            sorted(SVs.items(),
                   key=lambda item: item[1])[-int(len(SVs)*level):]
        ).keys()
    SVs = dict([(key, (SVs[key] if key in exposed_idx else 0))
                for key in SVs.keys()])
    return SVs


def privacy_protect(privacy_protection_measure, privacy_protection_level,
                    SV, SV_var=dict()):
    if privacy_protection_measure == 'DP':
        SV = differential_privacy(SV, privacy_protection_level)
    elif privacy_protection_measure == 'QT':
        SV = quantization(SV, privacy_protection_level)
    elif privacy_protection_measure == 'DR':
        for player_id in SV_var.keys():
            SV_var[player_id] = np.var(SV_var[player_id])
        SV = dimension_reduction(SV, privacy_protection_level, SV_var)
    return SV


class Output():
    def __init__(self,
                 convergence_threshold,
                 cache_size,
                 player_num,
                 privacy_protection_measure,
                 privacy_protection_level,
                 full_check_type):
        self.threshold = convergence_threshold
        self.cache_size = cache_size
        self.SV_cache = []
        self.player_num = player_num
        self.privacy_protection_measure = privacy_protection_measure
        self.privacy_protection_level = privacy_protection_level
        self.full_check_type = full_check_type

    def cache_SV(self, SVs):
        self.SV_cache.append(copy.deepcopy(SVs))
        return len(self.SV_cache) < self.cache_size

    def convergence_check(self,
                          calculated_num,
                          SVs,
                          SVs_var,
                          start_time,
                          utility_comp_num):
        if calculated_num == 0:
            return False

        if self.cache_SV(SVs):
            return False

        self.SVs = SVs
        self.SVs_var = SVs_var
        self.start_time = start_time
        self.utility_comp_num = utility_comp_num

        if self.full_check_type == 'coalition'  \
                and calculated_num >= 2**(self.player_num):
            print("Full sampling for all coalitions!")
            return True
        elif self.full_check_type == 'permutation'  \
                and calculated_num >= math.factorial(self.player_num):
            print("Full sampling for all permutations!")
            return True

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
            return self.privacy_protect()
        return self.result_output()

    def privacy_protect(self):
        # add privacy protection only when specified
        self.SVs = privacy_protect(self.privacy_protection_measure,
                                   self.privacy_protection_level,
                                   self.SVs, self.SVs_var)
        return self.result_output()

    def result_output(self):
        print(f"Final Resultant SVs: {self.SVs}")
        print(f"Final Resultant SV_var: {self.SVs_var}")
        print(f"Total runtime: {time.time()-self.start_time}")
        print(f"Total times of utility computation: {self.utility_comp_num}")
        return  \
            self.SVs, self.SVs_var, self.utility_comp_num
