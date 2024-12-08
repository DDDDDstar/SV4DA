import datetime
import pulp
import numpy as np
from scipy.optimize import minimize

from checker import Checker


class Output():
    def __init__(self,
                 convergence_threshold,
                 cache_size,
                 player_num,
                 privacy_protection_measure,
                 privacy_protection_level,
                 algo):
        self.aggregator = Aggregator(algo, player_num)
        self.checker = Checker(player_num,
                               cache_size, convergence_threshold)
        self.privacy = Privacy(
            measure=privacy_protection_measure,
            level=privacy_protection_level)
        self.task_total_utility = 0

    def result_process(self, results, full_sample, iter_times):
        self.aggregator.aggregate(results, iter_times, self.task_total_utility)
        if not full_sample:
            if not self.checker.convergence_check():
                print(f'Iteration {iter_times} done:')
                print(f"Current SV: {self.aggregator.SV}")
                print(
                    f"Current times of utility computation: {self.aggregator.utility_comp_times}")
                return False
        else:
            print("Full sample!")

        SV = self.aggregator.SV
        SV_var = self.aggregator.SV_var
        time_cost = self.aggregator.time_cost
        utility_comp_times = self.aggregator.utility_comp_times
        self.privacy.privacy_protect(
            SV, SV_var)
        print(f"Final Resultant SVs: {SV}")
        print(f"Final Resultant SV_var: {SV_var}")
        print(f"Total runtime: {time_cost}")
        print(
            f"Total times of utility computation: {utility_comp_times}")
        print(
            f"Average time cost of utility computation: {time_cost/utility_comp_times}")
        return True


class Aggregator():
    def __init__(self, algo, player_num):
        self.algo = algo
        self.player_num = player_num

        # SV settings
        self.SV = dict([(player_id, 0.0)
                        for player_id in range(self.player_num)])
        self.SV_var = dict([(player_id, [])
                            for player_id in range(self.player_num)])
        self.utility_comp_times = 0
        self.time_cost = 0

        if self.algo == 'MC':
            self.SV_comp_times = dict([(player_id, 0)
                                       for player_id in range(self.player_num)])

        if self.algo == 'GT':
            self.utilities = []

        if self.algo == 'RE':
            self.utilities = {0: 0}
            N = self.player_num
            self.A_RE = np.zeros((N, N))
            for i in range(N):
                for j in range(N):
                    if i == j:
                        self.A_RE[i, j] = sum([1/N/(N-k) for k in range(1, N)]) / \
                            sum([1/k/(N-k) for k in range(1, N)])
                    else:
                        self.A_RE[i, j] = 1/N/(N-1) *\
                            sum([(k-1)/(N-k) for k in range(2, N)]) / \
                            sum([1/k/(N-k) for k in range(1, N)])
            self.z_RE = np.array([0 for _ in range(N)]).reshape(1, -1)

        if self.algo == 'CP':
            self.num_measurement = int(self.player_num/2)
            self.y_CP = dict([(m, []) for m in range(self.num_measurement)])
            A_CP = np.random.binomial(1, 0.5, size=(self.num_measurement, N))
            self.A_CP = 1 / np.sqrt(self.num_measurement) * (2 * A_CP - 1)
            self.CP_epsilon = 0.00001

    def aggregate(self, results, iter_times, task_total_utility):
        if self.algo == 'MC':
            self.MC_aggregate(results)
        elif self.algo == 'MLE':
            self.MLE_aggregate(results)
        elif self.algo == 'GT':
            self.GT_aggregate(results, iter_times, task_total_utility)
        elif self.algo == 'CP':
            self.CP_aggregate(results, iter_times, task_total_utility)
        elif self.algo == 'RE':
            self.RE_aggregate(results, task_total_utility)

    def MC_aggregate(self, results):
        while not results.empty():
            (player_id, delta_utility, utility_comp_times, time_cost) = results.get()
            old_SV = self.SV[player_id]
            self.SV_comp_times[player_id] += 1
            self.SV[player_id]  \
                = (self.SV_comp_times[player_id]-1)*self.SV[player_id]+delta_utility/self.SV_comp_times[player_id]
            self.utility_comp_times += utility_comp_times
            self.time_cost += time_cost
            print(('[%s] Player %s: delta_utility: %s, SV_bef: %s, SV_aft: %s.') % (
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                player_id, delta_utility, old_SV, self.SV[player_id]))

    def MLE_aggregate(self, results):
        num_comp = np.zeros(self.player_num)
        e = np.zeros(self.player_num)
        while not results.empty():
            (player_id, delta_utility, utility_comp_times, time_cost) = results.get()
            if player_id != -1:
                e[player_id] += delta_utility
                num_comp[player_id] += 1
            self.utility_comp_times += utility_comp_times
            self.time_cost += time_cost

        # update SV
        self.SV = dict([(player_id, e[player_id] / num_comp[player_id])
                        for player_id in range(self.player_num)])
        for player_id in range(self.player_num):
            self.SV_var[player_id].append(self.SV[player_id])

    def GT_aggregate(self, results, iter_times, task_total_utility):
        while not results.empty():
            coalitions, value, utility_comp_times,  timeCost = results.get()
            self.utilities.append(([int(player_id in coalitions)
                                    for player_id in range(self.player_num)], value))
            if timeCost > 0:
                self.time_cost_per_utility_comp.append(timeCost)

        delta_utility = np.zeros((self.player_num, self.player_num))
        Z = 2 * sum([1/k for k in range(1, self.player_num)])
        for i in range(self.player_num):
            for j in range(i + 1, self.player_num):
                delta_utility[i, j] \
                    = Z/iter_times * sum([utility * (beta[i] - beta[j])
                                          for (beta, utility) in self.utilities])
                delta_utility[j, i] = - delta_utility[i, j]

        # find SV by solving the feasibility problem
        MyProbLP = pulp.LpProblem("LPProbDemo1", sense=pulp.LpMaximize)
        sv = [pulp.LpVariable('%s' % player_id, cat='Continuous')
              for player_id in range(self.player_num)]
        MyProbLP += sum(sv)
        for i in range(self.player_num):
            for j in range(i + 1, self.player_num):
                MyProbLP += (sv[i] - sv[j] - delta_utility[i, j]
                             <= self.args.GT_epsilon / 2 / np.sqrt(self.player_num))
                MyProbLP += (sv[i] - sv[j] - delta_utility[i, j]
                             >= - self.args.GT_epsilon / 2 / np.sqrt(self.player_num))

        print("feasible problem solving ...")
        MyProbLP += (sum(sv) >= task_total_utility)
        MyProbLP += (sum(sv) <= task_total_utility)
        MyProbLP.solve()
        # status： “Not Solved”, “Infeasible”,
        # “Unbounded”, “Undefined” or “Optimal”
        # print("Status:", pulp.LpStatus[MyProbLP.status])
        result = dict()
        for v in MyProbLP.variables():
            result[int(v.name)] = v.varValue
        # print('One solution for reference:', v.name, "=", v.varValue)
        # print("F(x) = ", pulp.value(MyProbLP.objective),
        #       self.task_total_utility)  # 输出最优解的目标函数值

        # update SV
        self.SV = dict([(player_id, result[player_id])
                        for player_id in range(self.player_num)])
        for player_id in range(self.player_num):
            self.SV_var[player_id].append(self.SV[player_id])

    def CP_aggregate(self, results, iter_times, task_total_utility):
        phi_t = dict()
        while not results.empty():
            (player_id, delta_utility, utility_comp_times, time_cost) = results.get()
            phi_t[player_id] = delta_utility
            self.utility_comp_times += utility_comp_times
            self.time_cost += time_cost
        for m in range(self.num_measurement):
            self.y_CP[m].append(
                sum([self.A_CP[m, player_id]*phi for (player_id, phi) in phi_t.items()]))
        y_mean = np.zeros(self.num_measurement)
        for m in range(len(self.y_CP)):
            y_mean[m] = 1/iter_times * sum(self.y_CP[m][:iter_times])
        sv_mean = task_total_utility/self.player_num
        def fun(sv_variance): return np.linalg.norm(sv_variance, ord=1)
        cons = (
            {'type': 'ineq', 'fun':
                lambda sv_variance:
                self.CP_epsilon - np.linalg.norm(
                    self.A_CP.dot(sv_mean+sv_variance)-y_mean, ord=2)},
        )
        res = minimize(fun, np.zeros(self.player_num),
                       method='SLSQP', constraints=cons)
        sv_variance = res.x

        # update SV
        self.SV = dict([(player_id, sv_mean+sv_variance[player_id])
                        for player_id in range(N)])
        for player_id in range(self.player_num):
            self.SV_var[player_id].append(self.SV[player_id])

    def RE_aggregate(self, results, task_total_utility):
        while not results.empty():
            order, value, utility_comp_times, timeCost = results.get()
            self.utilities[len(z)+order] = value
            self.utility_comp_times += utility_comp_times
            self.time_cost += timeCost

            # regression computation
            z = (np.concatenate((z, np.array(z_i))) if len(z_i) > 0 else z)
            b = np.zeros((self.player_num, 1))
            E_Z = 0.5*np.ones((self.player_num, 1))
            for (sample_id, z_i) in enumerate(z):
                b += (z_i.reshape(-1, 1) * self.utilities[sample_id]) / len(z)
            inv_A = np.linalg.inv(self.A_RE)
            ones = np.ones((self.player_num, 1))
            beta = np.linalg.inv(self.A_RE).dot(
                b-ones
                * ((ones.T.dot(inv_A).dot(b)-task_total_utility)
                   / ones.T.dot(inv_A).dot(ones))
            ).reshape(-1)

            # update SV
            self.SV = dict([(player_id, beta[player_id])
                            for player_id in range(self.player_num)])
            for player_id in range(self.player_num):
                self.SV_var[player_id].append(self.SV[player_id])


class Privacy():
    def __init__(self,  measure, level):
        self.measure = measure
        self.level = level

    def quantization(self, SVs):
        # level=0~1
        level = int(len(SVs)*self.level)
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
        return dict([(k, quantization_map[v]) for (k, v) in SVs.items()])

    def differential_privacy(self, SVs):
        noise = np.random.normal(0, self.level, len(SVs))
        return dict([(k, v + noise[idx])
                    for idx, (k, v) in enumerate(SVs.items())])

    def dimension_reduction(self, SVs, SVs_var):
        if len(SVs_var) == len(SVs):
            exposed_idx = dict(
                sorted(SVs_var.items(),
                       key=lambda item: item[1])[-int(len(SVs_var)*self.level):]
            ).keys()

        else:
            exposed_idx = dict(
                sorted(SVs.items(),
                       key=lambda item: item[1])[-int(len(SVs)*self.level):]
            ).keys()
        return dict([(key, (SVs[key] if key in exposed_idx else 0))
                    for key in SVs.keys()])

    def privacy_protect(self, SVs, SVs_var):
        if self.measure == 'DP':
            return self.differential_privacy(SVs)
        elif self.measure == 'QT':
            return self.quantization(SVs)
        elif self.measure == 'DR':
            for player_id in SVs_var.keys():
                SVs_var[player_id] = np.var(SVs_var[player_id])
            return self.dimension_reduction(SVs, SVs_var)
