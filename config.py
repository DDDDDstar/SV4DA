import threading
import sys
import importlib
import os
import numpy as np
import portalocker
import ast
import time

from Tasks import data_valuation, federated_learning, result_interpretation
from calculator import Shapley
from sampler import Sampler
from output import Output


OUT_PRINT_FLUSH_INTERVAL = 5
UTILITY_RECORD_FILEWRITE_INTERVAL = 10
BENCHMARK_ALGO = {
    'MC': ('permutation'),
    'MLE': ('coalition'),
    'GT': ('coalition'),
    'RE': ('permutation'),
    'CP': ('permutation')
}
BENCHMARK = {
    'DV': {
        'iris': (120),
        'wine': (142),
    },
    'FL': {
        'cifar': (10),
        'mnist': (10),
    },
    'RI': {
        'iris': (4),
        'wine': (13),
    },
    'DSV': {
        'iris': (12),
        'wine': (14),
    }
}


class Task():
    def __init__(self, args):
        self.args = args
        self.GA = args.GA
        self.TSS = args.TSS
        self.parallel_threads_num = 1 if self.GA else args.parallel_threads_num

        self.init_flag = True

        self.task = args.task
        self.dataset = args.dataset
        self.manual_seed = args.manual_seed
        self.utility_function = self.utility_computation_func_load(
            args.utility_function)
        if self.utility_function is None:
            self.init_flag = False
            return

        self.utility_record_file = args.utility_record_file
        self.utility_records = self.read_history_utility_record()
        if self.utility_records is None:
            self.init_flag = False
            return
        self.utility_record_write_lock = threading.Lock()
        self.utility_record_filewrite_lock = threading.Lock()
        self.utility_record_write_flag = False
        self.dirty_utility_record_num = 0

        if args.task in BENCHMARK:
            self.player_num = BENCHMARK[args.task][self.dataset]
            full_check_type = BENCHMARK_ALGO[args.algo][0]
        else:
            self.player_num = args.player_num
            full_check_type = args.full_check

        self.shapley = Shapley(
            player_num=self.player_num,
            utility_function=self.utility_computation_call,
            argorithm=args.algo,
            truncation=args.TC,
            truncation_threshold=args.TC_threshold,
            parallel_threads_num=self.parallel_threads_num,
            sampler=Sampler(
                sampling_strategy=args.sampling,
                algo=args.algo, player_num=self.player_num),
            output=Output(
                convergence_threshold=args.convergence_threshold,
                cache_size=args.SV_cache_size,
                player_num=self.player_num,
                full_check_type=full_check_type,
                privacy_protection_measure=args.privacy,
                privacy_protection_level=args.privacy_level))

        self.task_terminated = False
        self.flush_event = threading.Event()

    def utility_computation_func_load(self, utility_function_api):
        if self.task == 'DV':
            DV = data_valuation.DV(
                dataset=self.dataset,
                manual_seed=self.manual_seed,
                GA=self.GA, TSS=self.TSS)
            return DV.utility_computation
        elif self.task == 'FL':
            FL = federated_learning.FL(
                dataset=self.dataset,
                manual_seed=self.manual_seed,
                GA=self.GA, TSS=self.TSS)
            return FL.utility_computation
        # elif self.task == 'RI':
        #     return result_interpretation.utility_computation
        else:
            return utility_function_api

    def utility_computation_call(self, player_list):
        utility_record_idx = str(
            sorted(player_list) if self.task == 'FL' or (not self.GA) else player_list)
        if utility_record_idx in self.utility_records:
            return self.utility_records[utility_record_idx][0], -1

        start_time = time.time()
        try:
            utility = self.utility_function(player_list)
            time_cost = time.time() - start_time
            self.write_utility_record(utility_record_idx, utility, time_cost)
        except Exception as e:
            print(
                f"Call utility computation error:\n{e}")
            exit(-1)
        return utility, time_cost

    def printFlush(self):
        while not self.task_terminated:
            if self.flush_event.wait(OUT_PRINT_FLUSH_INTERVAL):
                sys.stdout.flush()

    def run_print_flush_thread(self):
        thread = threading.Thread(target=self.printFlush)
        thread.daemon = True
        thread.start()

    def pre_exp_statistic(self):
        utility_computation_timecost = dict()
        for player_idx in range(self.player_num):
            if self.task == 'DV':
                utility, time_cost = self.utility_computation_call(
                    range(player_idx))
                print(
                    f'Computing utility with {player_idx + 1} players tasks {time_cost} timeCost {utility} utility...')
                utility_computation_timecost[player_idx] = time_cost
        print('Average time cost for computing utility: ',
              np.mean(list(utility_computation_timecost.values())))

    def read_history_utility_record(self):
        self.utility_record_file = f'./Tasks/utility_records/{self.task}_{self.dataset}_{self.manual_seed}{"_GA" if self.GA else ""}{"_TSS" if self.TSS else ""}.log' \
            if self.utility_record_file == '' else self.utility_record_file

        if os.path.exists(self.utility_record_file):
            with portalocker.Lock(self.utility_record_file, 'r', encoding='utf-8', flags=portalocker.LOCK_SH) as file:
                utility_records = ast.literal_eval(file.read().strip())
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
                        self.utility_records.update(
                            ast.literal_eval(file.read().strip()))
                    self.dirty_utility_record_num = 0
                    ur = self.utility_records.copy()
                file.seek(0)
                file.write(str(ur))
            self.utility_record_filewrite_lock.release()

    def run(self):
        self.run_print_flush_thread()

        self.pre_exp_statistic()

        self.shapley.SV_calculate()
