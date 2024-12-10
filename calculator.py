import math
import time
import datetime
import threading
import queue
import pulp
import warnings
import portalocker
import os
import numpy as np

from config import UTILITY_RECORD_FILEWRITE_INTERVAL

warnings.filterwarnings('ignore')


class Shapley():
    def __init__(self, task, player_num, dataset,
                 utility_function, argorithm,
                 truncation, truncation_threshold,
                 parallel_threads_num, manual_seed,
                 GA, TSS,
                 utility_record_file,
                 sampler, output):
        self.task = task
        self.player_num = player_num
        self.utility_function = utility_function
        self.dataset = dataset
        self.GA = GA
        self.TSS = TSS
        self.manual_seed = manual_seed

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

        self.utility_record_file = utility_record_file
        self.utility_records = self.read_history_utility_record()
        if self.utility_records is None:
            print(
                f"ERROR: Read utility record failed: {utility_record_file}")
            exit(-1)

        self.utility_record_write_lock = threading.Lock()
        self.utility_record_filewrite_lock = threading.Lock()
        self.utility_record_write_flag = False
        self.dirty_utility_record_num = 0

    # check all threads and remove dead threads
    # return the number of alive threads
    def threads_clean(self):
        for t in self.threads:
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

    def utility_computation_call(self, player_list):
        utility_record_idx = str(
            sorted(player_list)
            if (self.task != 'DV' and self.task != 'DSV') or (not self.GA)
            else player_list)
        if utility_record_idx in self.utility_records:
            return self.utility_records[utility_record_idx]

        start_time = time.time()
        utility = self.utility_function(player_list)
        time_cost = time.time() - start_time
        self.write_utility_record(utility_record_idx, utility, time_cost)
        return utility, time_cost

    def read_history_utility_record(self):
        self.utility_record_file    \
            = f'./Tasks/utility_records/{self.task}_{self.dataset}_{self.manual_seed}{"_GA" if self.GA else ""}{"_TSS" if self.TSS else ""}.log' \
            if self.utility_record_file == '' else self.utility_record_file

        if os.path.exists(self.utility_record_file):
            with portalocker.Lock(self.utility_record_file, 'r', encoding='utf-8', flags=portalocker.LOCK_SH) as file:
                utility_records = eval(file.read().strip())
                # utility_records = json.load(file)
            return utility_records
        else:
            return {str([]): (0, 0)}

    def write_utility_record(self, utility_record_idx, utility, time_cost):
        if utility_record_idx in self.utility_records:
            return

        with self.utility_record_write_lock:
            self.utility_records[utility_record_idx] = (utility, time_cost)
            self.dirty_utility_record_num += 1

        if self.dirty_utility_record_num > UTILITY_RECORD_FILEWRITE_INTERVAL    \
                and self.utility_record_filewrite_lock.acquire(blocking=False):
            create = False
            if not os.path.exists(self.utility_record_file):
                os.mknod(self.utility_record_file)
                create = True
            with portalocker.Lock(self.utility_record_file, mode="r+", timeout=0) as file:
                with self.utility_record_write_lock:
                    if not create:
                        self.utility_records.update(eval(file.read().strip()))
                        # self.utility_records.update(json.load(file))
                    self.dirty_utility_record_num = 0
                    ur = self.utility_records.copy()
                file.seek(0)
                file.write(str(ur))
            self.utility_record_filewrite_lock.release()

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
        bef_addition, time_cost = self.utility_computation_call(subset)

        if self.if_truncation(bef_addition):
            aft_addition = bef_addition
            self.truncation_coaliations.add(
                ",".join(map(str, sorted(permutation[:order+1]))))
        else:
            # utility after adding the targeted player
            aft_addition, time_cost1 = self.utility_computation_call(
                permutation[:order+1])
            time_cost += time_cost1
            comp_times += 1
        results.put((player_id, aft_addition - bef_addition,
                     comp_times, time_cost))

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
        bef_addition, time_cost = self.utility_computation_call(subset)
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
            aft_addition, time_cost1 = self.utility_computation_call(
                list(subset)+[player_id])
            comp_times += 1
            time_cost += time_cost1
        results.put((player_id, aft_addition - bef_addition,
                     comp_times, time_cost))

    # refer to paper:
    # A Multilinear Sampling Algorithm to Estimate Shapley Values
    def MLE(self, **kwargs):
        MLE_M = 2
        MLE_interval = int(self.player_num/MLE_M)
        iter_num = MLE_interval + 1
        if self.sampler.sampling_strategy == 'antithetic':
            iter_num = int(MLE_interval/2) + 1
            MLE_M *= 2
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

    def GT_RE_parallelable_thread(self, selected_players, results):
        u, t = self.utility_computation_call(selected_players[:-1])
        if self.if_truncation(u):
            results.put(([int(player_id in selected_players)
                          for player_id in range(self.player_num)],
                         u, 1, t))
            self.truncation_coaliations.add(
                ",".join(map(str, sorted(selected_players))))
            return
        u, t = self.utility_computation_call(selected_players)
        results.put(([int(player_id in selected_players)
                      for player_id in range(self.player_num)],
                     u, 1, t))

    def GT(self, **kwargs):
        Z = 2 * sum([1/k for k in range(1, self.player_num)])
        q_k = [1/Z*(1/k+1/(self.player_num-k))
               for k in range(1, self.player_num)]

        results = queue.Queue()
        # sampling coalitions
        selected_coalitions = []
        selected_players = []
        for _ in range(self.player_num):
            selected_players, full_sample, iter_times = self.sampler.sample(
                q_k, selected_players)
            if full_sample:
                # use break here since
                # we need to process the case with len(selected_coalitions)>0
                break
            selected_coalitions.append(selected_players)

        if full_sample and len(selected_coalitions) == 0:
            # do not proceed to the following operations only when
            # full_sample=True and len(selected_coalitions) ==0 happens simultaneously
            return results, True, iter_times

        if self.parallel_threads_num == 1:
            for order, selected_players in enumerate(selected_coalitions):
                self.GT_RE_parallelable_thread(selected_players, results)
        else:
            # compute utility (speed up by multi-thread)
            for order, selected_players in enumerate(selected_coalitions):
                thread = threading.Thread(
                    target=self.GT_RE_parallelable_thread,
                    args=(selected_players, results))
                self.threads_controller('add', thread)
            self.threads_controller('finish')

        return results, False, iter_times

    def CP(self, **kwargs):
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

    def RE(self, **kwargs):
        results = queue.Queue()
        permutation, full_sample, iter_times = self.sampler.sample()
        if full_sample:
            return results, True, iter_times

        if self.parallel_threads_num == 1:
            for order, _ in enumerate(permutation):
                self.GT_RE_parallelable_thread(permutation[:order+1], results)
        else:
            for odd_even in [0, 1]:
                for order, _ in enumerate(permutation):
                    if order % 2 != odd_even:
                        continue
                    thread = threading.Thread(
                        target=self.GT_RE_parallelable_thread,
                        args=(permutation[:order+1], results))
                    self.threads_controller('add', thread)
            self.threads_controller('finish')
        return results, False, iter_times

    def SV_calculate(self):
        # print problem scale and the task's overall utility
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
        else:
            base_comp_func = self.argorithm

        # algorithm start
        results = None
        full_sample = False
        iter_times = 0
        while not self.output.result_process(results, full_sample, iter_times):
            results, full_sample, iter_times = base_comp_func() \
                if not callable(self.argorithm) \
                else self.argorithm(self.sampler.sample())

        return self.output.aggregator.SV, self.output.aggregator.SV_var
