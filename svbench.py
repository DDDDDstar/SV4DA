import threading
import sys
import os
import numpy as np
import portalocker
import time
from collections import namedtuple

from Tasks import data_valuation, federated_learning, result_interpretation, dataset_valuation
from calculator import Shapley
from sampler import Sampler
from output import Output
from config import config, BENCHMARK, OUT_PRINT_FLUSH_INTERVAL, UTILITY_RECORD_FILEWRITE_INTERVAL


class Task():
    def __init__(self, args):
        self.args = config(args)
        self.TC = ('TC' in args.optimization_strategy) if args.optimization_strategy is not None else False
        self.GA = ('GA' in args.optimization_strategy) if args.optimization_strategy is not None else False
        self.TSS = ('TSS' in args.optimization_strategy) if args.optimization_strategy is not None else False
        self.parallel_threads_num = 1 if self.GA else args.num_parallel_threads

        self.task = args.task
        self.dataset = args.dataset
        self.manual_seed = args.manual_seed
        self.utility_function = self.utility_computation_func_load(
            args.utility_function)
        if self.utility_function is None:
            print(
                f"ERROR: Get utility function failed: {args.utility_function}")
            exit(-1)

        self.utility_record_file = args.utility_record_file
        self.utility_records = self.read_history_utility_record()
        if self.utility_records is None:
            print(
                f"ERROR: Read utility record failed: {args.utility_record_file}")
            exit(-1)

        self.utility_record_write_lock = threading.Lock()
        self.utility_record_filewrite_lock = threading.Lock()
        self.utility_record_write_flag = False
        self.dirty_utility_record_num = 0

        if self.task in BENCHMARK:
            self.player_num = BENCHMARK[self.task][self.dataset]
        else:
            self.player_num = args.player_num

        self.shapley = Shapley(
            task=self.task,
            player_num=self.player_num,
            utility_function=self.utility_computation_call,
            argorithm=args.base_algo,
            truncation=self.TC,
            truncation_threshold=args.TC_threshold,
            parallel_threads_num=self.parallel_threads_num,
            sampler=Sampler(
                sampling=args.sampling_strategy,
                algo=args.base_algo, player_num=self.player_num),
            output=Output(
                convergence_threshold=args.convergence_threshold,
                cache_size=args.conv_check_num,
                player_num=self.player_num,
                privacy_protection_measure=args.privacy_protection_measure,
                privacy_protection_level=args.privacy_protection_level,
                algo=args.base_algo))

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
        elif self.task == 'RI':
            RI = result_interpretation.RI(
                dataset=self.dataset,
                manual_seed=self.manual_seed,
                GA=self.GA, TSS=self.TSS)
            return RI.utility_computation
        elif self.task == 'DSV':
            DSV = dataset_valuation.DSV(
                dataset=self.dataset,
                manual_seed=self.manual_seed,
                GA=self.GA, TSS=self.TSS)
            return DSV.utility_computation
        else:
            return utility_function_api

    def utility_computation_call(self, player_list):
        utility_record_idx = str(
            sorted(player_list) if (self.task != 'DV' and self.task != 'DSV') or (not self.GA) else player_list)
        if utility_record_idx in self.utility_records:
            return self.utility_records[utility_record_idx]

        start_time = time.time()
        utility = self.utility_function(player_list)
        time_cost = time.time() - start_time
        self.write_utility_record(utility_record_idx, utility, time_cost)
        return utility, time_cost

    def printFlush(self):
        while True:
            if self.flush_event.wait(OUT_PRINT_FLUSH_INTERVAL):
                sys.stdout.flush()

    def run_print_flush_thread(self):
        thread = threading.Thread(target=self.printFlush)
        thread.daemon = True
        thread.start()

    def pre_exp_statistic(self):
        utility_computation_timecost = dict()
        for player_idx in range(self.player_num):
            if self.task in ['DV', 'DSV']:
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

    def run(self):
        self.run_print_flush_thread()

        self.pre_exp_statistic()

        return self.shapley.SV_calculate()


def sv_calc(**kwargs):
    task = Task(args=kwargs)

    task.run()

    return task.run()
