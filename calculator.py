import math
import time
import datetime
import threading
import queue
import pulp
import warnings
import numpy as np

from scipy.optimize import minimize

warnings.filterwarnings('ignore')


class Shapley():
    def __init__(self, task, player_num,
                 utility_function,
                 argorithm,
                 truncation,
                 truncation_threshold,
                 parallel_threads_num,
                 sampler,
                 output):
        self.task = task
        self.player_num = player_num
        self.utility_function = utility_function

        # SV settings
        self.SV = dict([(player_id, 0.0)
                        for player_id in range(self.player_num)])
        self.SV_var = dict([(player_id, [])
                            for player_id in range(self.player_num)])

        # SV computation method's components
        self.argorithm = argorithm
        self.truncation_flag = truncation
        self.truncation_threshold = truncation_threshold
        self.parallel_threads_num = parallel_threads_num
        self.sampler = sampler

        # utility information
        self.task_total_utility = 0
        self.empty_set_utility = 0  # Only used for RE

        # runtime records
        self.start_time = 0
        self.utility_comp_num = 0
        self.time_cost_per_utility_comp = []
        self.truncation_coaliations = set()

        self.output = output

        self.threads = []

        self.CP_epsilon = 0.00001

    # check all threads and remove dead threads
    # return the number of alive threads
    def threads_clean(self):
        for t in self.threads[:]:
            if not t.is_alive():
                self.threads.remove(t)
        return len(self.threads)

    def threads_controller(self, op, thread=None):
        if op == 'add':
            if thread == None:
                raise Exception("Thread is None in thread addition op.")
            # if there are enough threads, wait for the first thread to finish
            # and remove it from the list
            if self.threads_clean() >= self.parallel_threads_num:
                self.threads[0].join()
                self.threads.pop(0)
            thread.daemon = True
            thread.start()
            self.threads.append(thread)
        elif op == 'finish':
            for thread in self.threads:
                thread.join()
            self.threads = []

    def if_truncation(self, bef_addition):
        utility_change_rate = np.abs((self.task_total_utility - bef_addition)
                                     / (self.task_total_utility + 10**(-15)))
        return False if not self.truncation_flag    \
            else utility_change_rate < self.truncation_threshold

    def iter_calculate(self, order, permutation, iter_times):
        startTime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        player_id = permutation[order]
        subset = permutation[:order]
        # utility before adding the targeted player
        bef_addition, time_cost = self.utility_function(subset)
        if time_cost > 0:
            self.time_cost_per_utility_comp.append(time_cost)

        if self.if_truncation(bef_addition):
            aft_addition = bef_addition
            self.truncation_coaliations.add(
                ",".join(map(str, sorted(permutation[:order+1]))))
        else:
            # utility after adding the targeted player
            aft_addition, time_cost = self.utility_function(
                permutation[:order+1])
            if time_cost > 0:
                self.time_cost_per_utility_comp.append(time_cost)
        # update SV
        old_SV = self.SV[player_id]
        self.SV[player_id] = ((iter_times - 1) * old_SV +
                              aft_addition - bef_addition) / iter_times
        self.SV_var[player_id].append(self.SV[player_id])

        # compute difference
        print(('[%s -- %s] Player %s at position %s/%s: utility_bef: %s, ' +
              'utility_aft: %s, SV_bef: %s, SV_aft: %s.') % (
                  startTime,
                  datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                  player_id, order, len(permutation),
                  bef_addition, aft_addition,
                  old_SV, self.SV[player_id]))
        return aft_addition - bef_addition

    def MC(self, **kwargs):
        permutation = kwargs.get('permutation')
        utility_calculated_coalitions = kwargs.get('coalitions')

        permutation, iter_times = self.sampler.sample(
            last=permutation)

        print('\n Monte Carlo iteration %s: ' % iter_times, permutation)
        if self.parallel_threads_num == 1:
            for idx, player_id in enumerate(permutation):
                self.iter_calculate(
                    idx, permutation, iter_times)

                utility_calculated_coalitions.add(
                    ",".join(map(str, sorted(permutation[:idx]))))
                utility_calculated_coalitions.add(
                    ",".join(map(str, sorted(permutation[:idx+1]))))
        else:
            for odd_even in [0, 1]:
                for idx, player_id in enumerate(permutation):
                    if idx % 2 != odd_even:
                        continue
                    thread = threading.Thread(
                        target=self.iter_calculate,
                        args=(idx, permutation, iter_times))

                    self.threads_controller('add', thread)
                    utility_calculated_coalitions.add(
                        ",".join(map(str, sorted(permutation[:idx]))))
                    utility_calculated_coalitions.add(
                        ",".join(map(str, sorted(permutation[:idx+1]))))
                self.threads_controller('finish')

        # print current progress
        truncation_num = len(self.truncation_coaliations)
        self.utility_comp_num = \
            len(utility_calculated_coalitions) - truncation_num
        print(f'Monte Carlo iteration {iter_times} done:')
        print(f"Current SV: {self.SV}")
        print(f"Current runtime: {time.time()-self.start_time} s")
        if self.truncation_flag:
            print(f"Current number of truncations: {truncation_num}")
        print(
            f"Current times of utility computation: {self.utility_comp_num}")

        return iter_times

    def MLE_parallelable_thread(self,
                                player_id,
                                I_mq,
                                results):
        subset = [player_id_
                  for player_id_ in range(self.player_num)
                  if I_mq[player_id_] == 1]
        # utility before adding the targeted player
        bef_addition, time_cost = self.utility_function(subset)
        results.put((-1, -1, time_cost))

        if player_id in subset:
            results.put((player_id, 0, 0))
            return
        if self.if_truncation(bef_addition):
            time_cost = 0
            aft_addition = bef_addition
            self.truncation_coaliations.add(
                ",".join(map(str, sorted(list(subset)+[player_id]))))
        else:
            # utility after adding the targeted player
            aft_addition, time_cost = self.utility_function(
                list(subset)+[player_id])
        results.put((player_id, aft_addition-bef_addition, time_cost))

    # refer to paper:
    # A Multilinear Sampling Algorithm to Estimate Shapley Values
    def MLE(self, **kwargs):
        MLE_interval = kwargs.get('MLE_interval')
        MLE_M = kwargs.get('MLE_M')
        iter_num = int(MLE_interval / 2) + 1  \
            if self.sampler.sampling_strategy == 'antithetic'   \
            else MLE_interval + 1
        e = np.zeros(self.player_num)
        num_comp = np.zeros(self.player_num)

        print(
            f'MLE iteration(with interval_{MLE_interval}) start!')
        results = queue.Queue()
        I_mq = []
        full_sampling_flag = False
        for iter_ in range(iter_num):
            if full_sampling_flag:
                break
            for m in range(MLE_M):
                I_mq, full_sampling_flag = self.sampler.sample(
                    q=iter_ / MLE_interval, I_mq=I_mq, m=m)
                if full_sampling_flag:
                    break

                if self.parallel_threads_num == 1:
                    for player_id in range(self.player_num):
                        self.MLE_parallelable_thread(
                            player_id, I_mq, results)
                else:
                    # speed up by multiple threads
                    for player_id in range(self.player_num):
                        # compute under the other q values
                        thread = threading.Thread(
                            target=self.MLE_parallelable_thread,
                            args=(player_id, I_mq, results))
                        self.threads_controller('add', thread)
                    self.threads_controller('finish')

        while not results.empty():
            (player_id, delta_utility, time_cost) = results.get()
            if player_id != -1:
                e[player_id] += delta_utility
                num_comp[player_id] += 1
            if time_cost > 0:
                self.time_cost_per_utility_comp.append(time_cost)

        # update SV
        self.SV = dict([(player_id, e[player_id] / num_comp[player_id])
                        for player_id in range(self.player_num)])
        for player_id in range(self.player_num):
            self.SV_var[player_id].append(self.SV[player_id])

        # print current progress
        self.utility_comp_num   \
            = len(self.sampler.coalitions)-len(self.truncation_coaliations)
        print('MLE iteration (with MLE_interval_%s) done' % MLE_interval)
        print("Current SV: ", self.SV)
        print("Current runtime: ", time.time()-self.startTime)
        if self.truncation_flag:
            print("Current number of truncations: ",
                  len(self.truncation_coaliations))
        print("Current times of utility computation: ",
              self.utility_comp_num)
        return len(self.sampler.coalitions)

    def GT_RE_parallelable_thread(self, order, selected_players, results):
        u, t = self.utility_function(selected_players[-1])
        if self.if_truncation(u):
            results.put((order, (u, 0)))
            self.truncation_coaliations.add(
                ",".join(map(str, sorted(selected_players))))
            return
        u, t = self.utility_function(selected_players)
        results.put((order, (u, t)))

    def GT(self, **kwargs):
        Z = kwargs.get('Z')
        q_k = kwargs.get('q_k')
        utilities = kwargs.get('utilities')

        # sampling coalitions
        selected_coalitions = []
        selected_players = []
        for _ in range(self.player_num):
            selected_players, full_sampling_flag, iter_times = self.sampler.sample(
                q_k, selected_players)
            selected_coalitions.append(selected_players)
            if full_sampling_flag:
                break

        results = queue.Queue()
        if self.parallel_threads_num == 1:
            for order, selected_players in enumerate(selected_coalitions):
                self.GT_RE_parallelable_thread(
                    order, selected_players, results)
        else:
            # compute utility (speed up by multi-thread)
            for order, selected_players in enumerate(selected_coalitions):
                thread = threading.Thread(
                    target=self.GT_RE_parallelable_thread,
                    args=(order, selected_players, results))
                self.threads_controller('add', thread)
            self.threads_controller('finish')

        while not results.empty():
            order, (value, timeCost) = results.get()
            utilities.append(([int(player_id in selected_coalitions[order])
                               for player_id in range(self.player_num)], value))
            if timeCost > 0:
                self.time_cost_per_utility_comp.append(timeCost)

        print('Group testing iteration %s with (k=%s): ' % (
            iter_times,
            [len(selected_players) for selected_players in selected_coalitions]))
        delta_utility = np.zeros((self.player_num, self.player_num))
        for i in range(self.player_num):
            for j in range(i + 1, self.player_num):
                delta_utility[i, j] \
                    = Z/iter_times * sum([utility * (beta[i] - beta[j])
                                          for (beta, utility) in utilities])
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
        MyProbLP += (sum(sv) >= self.task_total_utility)
        MyProbLP += (sum(sv) <= self.task_total_utility)
        MyProbLP.solve()
        # status： “Not Solved”, “Infeasible”,
        # “Unbounded”, “Undefined” or “Optimal”
        print("Status:", pulp.LpStatus[MyProbLP.status])
        result = dict()
        for v in MyProbLP.variables():
            result[int(v.name)] = v.varValue
        print('One solution for reference:', v.name, "=", v.varValue)
        print("F(x) = ", pulp.value(MyProbLP.objective),
              self.task_total_utility)  # 输出最优解的目标函数值

        # update SV
        self.SV = dict([(player_id, result[player_id])
                        for player_id in range(self.player_num)])
        for player_id in range(self.player_num):
            self.SV_var[player_id].append(self.SV[player_id])

        # print current progress
        print('Group testing iteration %s done (len utilities: %s)!' % (
            iter_times, len(utilities)))
        print("Current SV: ", self.SV)
        print("Current runtime: ", time.time()-self.start_time)
        print("Current number of truncations: ",
              len(self.truncation_coaliations))
        print("Current times of utility computation: ",
              len(self.sampler.coalitions)-len(self.truncation_coaliations))
        print("Average time cost of a single time of utility computation: ",
              (np.average(self.time_cost_per_utility_comp)
               if len(self.time_cost_per_utility_comp) > 0 else 0))
        return len(self.sampler.coalitions)

    def CP(self, **kwargs):
        permutation = kwargs.get('permutation')
        utility_calculated_coalitions = kwargs.get('coalitions')
        A = kwargs.get('A_CP')
        y = kwargs.get('y_CP')

        N = self.player_num
        num_measurement = int(N/2)

        permutation, iter_times = self.sampler.sample(
            last=permutation)
        print('\n Compressive permutation sampling iteration %s: ' %
              iter_times, permutation)
        phi_t = dict()
        if self.parallel_threads_num == 1:
            for idx, player_id in enumerate(permutation):
                phi_t[player_id] = self.iter_calculate(
                    idx, permutation, iter_times)
                utility_calculated_coalitions.add(
                    ",".join(map(str, sorted(permutation[:idx]))))
                utility_calculated_coalitions.add(
                    ",".join(map(str, sorted(permutation[:idx+1]))))
        else:
            for odd_even in [0, 1]:
                for idx, player_id in enumerate(permutation):
                    if idx % 2 != odd_even:
                        continue
                    thread = threading.Thread(
                        target=self.iter_calculate,
                        args=(idx, permutation, iter_times)
                    )
                    self.threads_controller('add', thread)
                    utility_calculated_coalitions.add(
                        ",".join(map(str, sorted(permutation[:idx]))))
                    utility_calculated_coalitions.add(
                        ",".join(map(str, sorted(permutation[:idx+1]))))
                self.threads_controller('finish')

        for m in range(num_measurement):
            y[m].append(
                sum([A[m, player_id]*phi for (player_id, phi) in phi_t.items()]))

        y_mean = np.zeros(num_measurement)
        for m in range(len(y)):
            y_mean[m] = 1/iter_times * sum(y[m][:iter_times])
        sv_mean = self.task_total_utility/N
        def fun(sv_variance): return np.linalg.norm(sv_variance, ord=1)
        cons = (
            {'type': 'ineq', 'fun':
                lambda sv_variance:
                self.CP_epsilon - np.linalg.norm(
                    A.dot(sv_mean+sv_variance)-y_mean, ord=2)},
        )
        res = minimize(fun, np.zeros(N), method='SLSQP', constraints=cons)
        sv_variance = res.x

        # update SV
        self.SV = dict([(player_id, sv_mean+sv_variance[player_id])
                        for player_id in range(N)])
        for player_id in range(N):
            self.SV_var[player_id].append(self.SV[player_id])

        # print current progress
        print('CPS iteration %s done (len y: %s)!' %
              (iter_times, len(y[0])))
        print("Current SV: ", self.SV)
        print("Current runtime: ", time.time()-self.start_time)
        print("Current number of truncations: ",
              len(self.truncation_coaliations))
        print("Current times of utility computation: ",
              len(utility_calculated_coalitions)-len(self.truncation_coaliations))
        print("Average time cost of a single time of utility computation: ",
              (np.average(self.time_cost_per_utility_comp)
               if len(self.time_cost_per_utility_comp) > 0 else 0))
        return iter_times

    def RE(self, **kwargs):
        permutation = kwargs.get('permutation')
        utility_calculated_coalitions = kwargs.get('coalitions')
        A = kwargs.get('A_RE')
        z = kwargs.get('z_RE')
        utilities = kwargs.get('utilities_RE')

        permutation, iter_times = self.sampler.sample(last=permutation)
        z_i = []
        results = queue.Queue()
        for order, _ in enumerate(permutation):
            if ",".join(map(str, sorted(permutation[:order+1]))) in utility_calculated_coalitions:
                continue

            z_i.append([int(player_id in permutation[:order+1])
                        for player_id in range(self.player_num)])
            if self.parallel_threads_num == 1:
                self.GT_RE_parallelable_thread(
                    len(z_i)-1, permutation[:order+1], results)
            else:
                thread = threading.Thread(
                    target=self.GT_RE_parallelable_thread,
                    args=(len(z_i)-1, permutation[:order+1], results))
                self.threads_controller('add', thread)
            utility_calculated_coalitions.add(
                ",".join(map(str, sorted(permutation[:order]))))
            utility_calculated_coalitions.add(
                ",".join(map(str, sorted(permutation[:order+1]))))
        self.threads_controller('finish')

        while not results.empty():
            order, (value, timeCost) = results.get()
            utilities[len(z)+order] = value
            if timeCost > 0:
                self.time_cost_per_utility_comp.append(timeCost)

            # regression computation
            z = (np.concatenate((z, np.array(z_i))) if len(z_i) > 0 else z)
            b = np.zeros((self.player_num, 1))
            E_Z = 0.5*np.ones((self.player_num, 1))
            for (sample_id, z_i) in enumerate(z):
                b += (z_i.reshape(-1, 1) *
                      utilities[sample_id] - E_Z * self.empty_set_utility) / len(z)
            inv_A = np.linalg.inv(A)
            ones = np.ones((self.player_num, 1))
            beta = np.linalg.inv(A).dot(
                b-ones
                * ((ones.T.dot(inv_A).dot(b)-self.task_total_utility + self.empty_set_utility)
                   / ones.T.dot(inv_A).dot(ones))
            ).reshape(-1)

            # update SV
            self.SV = dict([(player_id, beta[player_id])
                            for player_id in range(self.player_num)])
            for player_id in range(self.player_num):
                self.SV_var[player_id].append(self.SV[player_id])

            # print current progress
            print('Regression iteration %s done (len z:%s)' % (
                iter_times, len(z)))
            print("Current SV: ", self.SV)
            print("Current runtime: ", time.time()-self.start_time)
            print("Current number of truncations: ",
                  len(self.truncation_coaliations))
            print("Current times of utility computation: ",
                  len(utility_calculated_coalitions)-len(self.truncation_coaliations))

            return iter_times

    def problemScale_statistics(self):
        print('【Problem Scale of SV Exact Computation】')
        print('Total number of players: ', self.player_num)
        print('(coalition sampling) Total number of utility computations: ',
              '%e' % (2*self.player_num * 2**(self.player_num-1)))
        print('(permutation sampling) Total number of utility computations:',
              '%e' % (2*self.player_num * math.factorial(self.player_num)))
        self.task_total_utility, _ = self.utility_function(
            range(self.player_num))
        print('The task\'s total utility: ', self.task_total_utility)

    def SV_calculate(self):
        # print problem scale and the task's overall utility
        self.problemScale_statistics()
        if not callable(self.argorithm):
            if self.argorithm == 'MC':
                base_comp_func = self.MC
            elif self.argorithm == 'MLE':
                base_comp_func = self.MLE
            elif self.argorithm == 'RE':
                base_comp_func = self.RE
                self.empty_set_utility, _ = self.utility_function([])
                print('The RE task\'s emptySet utility: ',
                      self.empty_set_utility)
            elif self.argorithm == 'GT':
                base_comp_func = self.GT
            elif self.argorithm == 'CP':
                base_comp_func = self.CP

        self.start_time = time.time()
        avg_time_cost = 0
        calculated_num = 0
        resultant_SVs = []
        resultant_SVs_var = []
        utility_comp_num = 0

        N = self.player_num
        # MC & CP & RE paras
        permutation = list(range(N))
        coalitions = set()
        # CP paras
        num_measurement = int(N/2)
        A_CP = np.random.binomial(1, 0.5, size=(num_measurement, N))
        A_CP = 1 / np.sqrt(num_measurement) * (2 * A_CP - 1)
        y_CP = dict([(m, []) for m in range(num_measurement)])
        # MLE paras
        MLE_interval = 0
        MLE_M = 2
        # GT paras
        Z = 2 * sum([1/k for k in range(1, N)])
        q_k = [1/Z*(1/k+1/(N-k))
               for k in range(1, N)]
        utilities_GT = []
        # RE paras
        A_RE = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if i == j:
                    A_RE[i, j] = sum([1/N/(N-k) for k in range(1, N)]) / \
                        sum([1/k/(N-k) for k in range(1, N)])
                else:
                    A_RE[i, j] = 1/N/(N-1) *\
                        sum([(k-1)/(N-k) for k in range(2, N)]) / \
                        sum([1/k/(N-k) for k in range(1, N)])
        z_RE = np.array([0 for _ in range(N)]).reshape(1, -1)
        utilities_RE = {0: self.empty_set_utility}
        while not self.output.convergence_check(calculated_num=calculated_num,
                                                SVs=resultant_SVs,
                                                SVs_var=resultant_SVs_var,
                                                start_time=self.start_time,
                                                utility_comp_num=utility_comp_num):
            if self.argorithm == 'MLE':
                MLE_interval += int(self.player_num/MLE_M)
                if self.sampler.sampling_strategy == 'antithetic':
                    MLE_M *= 2
            if not callable(self.argorithm):
                calculated_num = base_comp_func(
                    permutation=permutation, coalitions=coalitions,
                    A_CP=A_CP, y_CP=y_CP,
                    MLE_interval=MLE_interval, MLE_M=MLE_M,
                    Z=Z, q_k=q_k, utilities_GT=utilities_GT,
                    A_RE=A_RE, z_RE=z_RE, utilities_RE=utilities_RE)
                resultant_SVs = self.SV
                resultant_SVs_var = self.SV_var
                utility_comp_num = self.utility_comp_num
                avg_time_cost = np.average(self.time_cost_per_utility_comp) if len(
                    self.time_cost_per_utility_comp) > 0 else 0
            else:
                resultant_SVs, resultant_SVs_var = self.argorithm(
                    self.sampler.sample())
                calculated_num += 1
