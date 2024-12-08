import math
import time
import datetime
import threading
import queue
import pulp
import warnings
import numpy as np

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

        # SV computation method's components
        self.argorithm = argorithm
        self.truncation_flag = truncation
        self.truncation_threshold = truncation_threshold
        self.parallel_threads_num = parallel_threads_num
        self.sampler = sampler
        self.output = output

        self.task_total_utility = 0
        self.CP_epsilon = 0.00001

        self.truncation_coaliations = set()
        self.threads = []

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

    def MC_CP_parallelable_thread(self, order, permutation, results):
        comp_times = 1
        player_id = permutation[order]
        subset = permutation[:order]
        # utility before adding the targeted player
        bef_addition, time_cost = self.utility_function(subset)

        if self.if_truncation(bef_addition):
            aft_addition = bef_addition
            self.truncation_coaliations.add(
                ",".join(map(str, sorted(permutation[:order+1]))))
        else:
            # utility after adding the targeted player
            aft_addition, time_cost1 = self.utility_function(
                permutation[:order+1])
            time_cost += time_cost1
            comp_times += 1
        results.put((player_id, aft_addition - bef_addition, comp_times, time_cost))

    def MC(self, **kwargs):
        permutation, full_sample, iter_times = self.sampler.sample()

        results = queue.Queue()
        if full_sample:
            return results, True, iter_times

        print('\n Monte Carlo iteration %s: ' % iter_times, permutation)
        if self.parallel_threads_num == 1:
            for idx, player_id in enumerate(permutation):
                self.MC_CP_parallelable_thread(
                    idx, permutation, results)
        else:
            for odd_even in [0, 1]:
                for idx, player_id in enumerate(permutation):
                    if idx % 2 != odd_even:
                        continue
                    thread = threading.Thread(
                        target=self.MC_CP_parallelable_thread,
                        args=(idx, permutation, results))

                    self.threads_controller('add', thread)

                self.threads_controller('finish')

        return results, False, iter_times

    def MLE_parallelable_thread(self,
                                player_id,
                                I_mq,
                                results):
        subset = [player_id_
                  for player_id_ in range(self.player_num)
                  if I_mq[player_id_] == 1]
        # utility before adding the targeted player
        bef_addition, time_cost = self.utility_function(subset)
        comp_times = 1

        if player_id in subset:
            results.put((player_id, 0, 0, 0))
            return
        if self.if_truncation(bef_addition):
            aft_addition = bef_addition
            self.truncation_coaliations.add(
                ",".join(map(str, sorted(list(subset)+[player_id]))))
        else:
            # utility after adding the targeted player
            aft_addition, time_cost1 = self.utility_function(
                list(subset)+[player_id])
            comp_times += 1
            time_cost += time_cost1
        results.put((player_id, aft_addition -
                    bef_addition, comp_times, time_cost))

    # refer to paper:
    # A Multilinear Sampling Algorithm to Estimate Shapley Values
    def MLE(self, **kwargs):
        MLE_interval = self.MLE_interval
        MLE_M = self.MLE_M
        iter_num = int(MLE_interval / 2) + 1  \
            if self.sampler.sampling_strategy == 'antithetic'   \
            else MLE_interval + 1
        print(
            f'MLE iteration(with interval_{MLE_interval}) start!')
        results = queue.Queue()
        I_mq = []
        for iter_ in range(iter_num):
            for m in range(MLE_M):
                I_mq, full_sample, iter_times = self.sampler.sample(
                    q=iter_ / MLE_interval, I_mq=I_mq, m=m)
                if full_sample:
                    return results, True, iter_times

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

        return results, False, iter_times

    def GT_parallelable_thread(self, player_id, selected_players, results):
        u, t = self.utility_function(selected_players[-1])
        comp_times = 1
        if self.if_truncation(u):
            results.put((player_id, u, comp_times, 0))
            self.truncation_coaliations.add(
                ",".join(map(str, sorted(selected_players))))
            return
        u, t1 = self.utility_function(selected_players)
        comp_times += 1
        t += t1
        results.put((player_id, u, comp_times, t))

    def GT(self, **kwargs):
        q_k = kwargs.get('q_k')
        results = queue.Queue()

        # sampling coalitions
        selected_coalitions = []
        selected_players = []
        for _ in range(self.player_num):
            selected_players, full_sample, iter_times = self.sampler.sample(
                q_k, selected_players)
            if full_sample:
                return results, True, iter_times
            selected_coalitions.append(selected_players)

        if self.parallel_threads_num == 1:
            for order, selected_players in enumerate(selected_coalitions):
                self.GT_parallelable_thread(
                    selected_coalitions[order], selected_players, results)
        else:
            # compute utility (speed up by multi-thread)
            for order, selected_players in enumerate(selected_coalitions):
                thread = threading.Thread(
                    target=self.GT_parallelable_thread,
                    args=(selected_coalitions[order], selected_players, results))
                self.threads_controller('add', thread)
            self.threads_controller('finish')

        return results, False, iter_times

    def CP(self, **kwargs):
        N = self.player_num
        num_measurement = int(N/2)

        phi_t = queue.Queue()
        permutation, full_sample, iter_times = self.sampler.sample()
        if full_sample:
            return phi_t, True, iter_times
        print('\n Compressive permutation sampling iteration %s: ' %
              iter_times, permutation)
        if self.parallel_threads_num == 1:
            for idx, player_id in enumerate(permutation):
                self.MC_CP_parallelable_thread(
                    idx, permutation, iter_times, phi_t)
        else:
            for odd_even in [0, 1]:
                for idx, player_id in enumerate(permutation):
                    if idx % 2 != odd_even:
                        continue
                    thread = threading.Thread(
                        target=self.MC_CP_parallelable_thread,
                        args=(idx, permutation, iter_times, phi_t)
                    )
                    self.threads_controller('add', thread)
                self.threads_controller('finish')
        return phi_t, False, iter_times

    def RE_parallelable_thread(self, order, selected_players, results):
        player_id = selected_players[order]
        u, t = self.utility_function(selected_players[-1])
        comp_times = 1
        if self.if_truncation(u):
            results.put((order, u, comp_times, 0))
            self.truncation_coaliations.add(
                ",".join(map(str, sorted(selected_players))))
            return
        u, t1 = self.utility_function(selected_players)
        comp_times += 1
        t += t1
        results.put((order, u, comp_times, t))

    def RE(self, **kwargs):
        utility_calculated_coalitions = kwargs.get('coalitions')

        results = queue.Queue()
        permutation, full_sample, iter_times = self.sampler.sample()
        if full_sample:
            return results, True, iter_times
        z_i = []
        for order, _ in enumerate(permutation):
            if ",".join(map(str, sorted(permutation[:order+1]))) in utility_calculated_coalitions:
                continue

            z_i.append([int(player_id in permutation[:order+1])
                        for player_id in range(self.player_num)])
            if self.parallel_threads_num == 1:
                self.RE_parallelable_thread(
                    len(z_i)-1, permutation[:order+1], results)
            else:
                thread = threading.Thread(
                    target=self.RE_parallelable_thread,
                    args=(len(z_i)-1, permutation[:order+1], results))
                self.threads_controller('add', thread)
            utility_calculated_coalitions.add(
                ",".join(map(str, sorted(permutation[:order]))))
            utility_calculated_coalitions.add(
                ",".join(map(str, sorted(permutation[:order+1]))))
        self.threads_controller('finish')
        return results, False, iter_times

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
        self.output.task_total_utility = self.task_total_utility

    def SV_calculate(self):
        # print problem scale and the task's overall utility
        self.problemScale_statistics()
        if not callable(self.argorithm):
            if self.argorithm == 'MC':
                base_comp_func = self.MC
            elif self.argorithm == 'MLE':
                base_comp_func = self.MLE
                self.MLE_interval = 0
                self.MLE_M = 2
            elif self.argorithm == 'RE':
                base_comp_func = self.RE
            elif self.argorithm == 'GT':
                base_comp_func = self.GT
            elif self.argorithm == 'CP':
                base_comp_func = self.CP

        N = self.player_num
        # RE paras
        coalitions = set()
        # GT paras
        Z = 2 * sum([1/k for k in range(1, self.player_num)])
        q_k = [1/Z*(1/k+1/(N-k))
               for k in range(1, N)]
        flag = True
        while flag or not self.output.result_process(results, full_sample, iter_times):
            flag = False
            if self.argorithm == 'MLE':
                self.MLE_interval += int(self.player_num/self.MLE_M)
                if self.sampler.sampling_strategy == 'antithetic':
                    self.MLE_M *= 2

            results, full_sample, iter_times = base_comp_func(
                coalitions=coalitions, q_k=q_k) if not callable(self.argorithm)    \
                else self.argorithm(self.sampler.sample())

        return self.output.aggregator.SV, self.output.aggregator.SV_var
