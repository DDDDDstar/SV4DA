import threading
import math
import time
import sys
from Tasks import data_valuation, federated_learning, result_interpretation, dataset_valuation
from calculator import Shapley
from sampler import Sampler
from output import Output
from config import config, BENCHMARK#, OUT_PRINT_FLUSH_INTERVAL


class SVBench():
    def __init__(self, args):
        self.args = config(args)
        optimization_strategy = args.get('optimization_strategy') if args.get(
            'optimization_strategy') is not None else ''
        self.TC = 'TC' in optimization_strategy
        self.GA = 'GA' in optimization_strategy
        self.TSS = 'TSS' in optimization_strategy
        self.parallel_threads_num = args.get('num_parallel_threads')

        self.task = args.get('task')
        self.dataset = args.get('dataset')
        self.manual_seed = args.get('manual_seed')
        self.utility_function = self.utility_computation_func_load(
            args.get('utility_function'))
        if self.utility_function is None:
            print(
                f"ERROR: Get utility function failed: {args.get('utility_function')}")
            exit(-1)

        if self.task in BENCHMARK:
            self.player_num = BENCHMARK[self.task][self.dataset]
        else:
            self.player_num = args.get('player_num')

        base_algo = args.get('base_algo')
        self.shapley = Shapley(
            task=self.task,
            player_num=self.player_num,
            utility_function=self.utility_function,
            argorithm=base_algo,
            truncation=self.TC,
            truncation_threshold=args.get('TC_threshold'),
            parallel_threads_num=self.parallel_threads_num,
            dataset=self.dataset,
            GA=self.GA,
            TSS=self.TSS,
            manual_seed=self.manual_seed,
            utility_record_file=args.get('utility_record_file'),
            sampler=Sampler(
                sampling=args.get('sampling_strategy'),
                algo=base_algo, player_num=self.player_num),
            output=Output(
                convergence_threshold=args.get('convergence_threshold'),
                cache_size=args.get('conv_check_num'),
                player_num=self.player_num,
                privacy_protection_measure=args.get(
                    'privacy_protection_measure'),
                privacy_protection_level=args.get(
                    'privacy_protection_level'),
                task_total_utility=self.utility_function(
                    range(self.player_num)),
                task_emptySet_utility=self.utility_function([]),
                algo=base_algo))

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
    '''
    def printFlush(self):
        while True:
            time.sleep(OUT_PRINT_FLUSH_INTERVAL)
            sys.stdout.flush()

    def run_print_flush_thread(self):
        thread = threading.Thread(target=self.printFlush)
        thread.daemon = True
        thread.start()
    '''
    def pre_exp_statistic(self):
        print('【Problem Scale of SV Computation】')
        print('Total number of players: ', self.player_num)
        print('(coalition sampling) Total number of utility computations: ',
              '%e' % (2*self.player_num * 2**(self.player_num-1)))
        print('(permutation sampling) Total number of utility computations:',
              '%e' % (2*self.player_num * math.factorial(self.player_num)))

        utility_computation_timecost = dict()
        for player_idx in range(self.player_num):
            start_time = time.time()
            utility = self.utility_function(range(player_idx))
            time_cost = time.time() - start_time
            print(
                f'Computing utility with {player_idx + 1} players tasks {time_cost} timeCost {utility} utility...')
            utility_computation_timecost[player_idx] = time_cost

        print('Average time cost for computing utility:',
              sum(utility_computation_timecost.values())/len(utility_computation_timecost))
        print('The task\'s total utility: ', utility)

    def run(self):
        #self.run_print_flush_thread()
        self.pre_exp_statistic()
        computing_results = self.shapley.SV_calculate()
        sys.stdout.flush()
        return computing_results


def sv_calc(**kwargs):
    computing_algo = SVBench(args=kwargs)

    return computing_algo.run()
