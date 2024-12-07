import sys

from collections import namedtuple

from config import Task, BENCHMARK


def para_set(args):
    required = ['task']
    # only required for bench tasks:
    bench_required = ['dataset']
    # only required for user-specific tasks:
    user_required = ['utility_function', 'full_check', 'player_num']
    paras = {'log_file': 'std',
             'utility_record_file': '',
             'optimization_strategy': None,
             'TC_threshold': 0.01,
             'conv_check_num': 5,
             'base_algo': 'MC',
             'sampling_strategy': 'random',
             'convergence_threshold': 0.1,
             'num_parallel_threads': 1,
             'manual_seed': 42,
             'privacy_protection_measure': None,
             'privacy_protection_level': 0.5,
             'utility_function': None
             }
    for required_key in required:
        if required_key not in args:
            print(f'Missing required argument: {required_key}')
            return -1
    if args.get('task') in BENCHMARK:
        for bench_key in bench_required:
            if bench_key not in args:
                print(
                    f'Missing required argument for benchmark task: {bench_key}')
                return -1
    else:
        for user_key in user_required:
            if user_key not in args:
                print(
                    f'Missing required argument for user-specific task: {user_key}')
                return -1

    args.update({key: value for key, value in paras.items() if key not in args})
    return 0


def open_log_file(log_file):
    if log_file != 'std':
        try:
            file = open(log_file, 'w')
            sys.stdout = file
        except Exception as e:
            print(f"Open log file error:\n{e}")
            return -1
    return 0


def sv_calc(**kwargs):
    if para_set(kwargs) == -1:
        exit(-1)
    print(f'Experiment arguments:\n{kwargs}')
    ARGS = namedtuple('ARGS', kwargs.keys())
    kwargs = ARGS(**kwargs)

    if open_log_file(kwargs.log_file) == -1:
        exit(-1)

    task = Task(args=kwargs)
    if not task.init_flag:
        exit(-1)

    task.run()


if __name__ == '__main__':
    # sv_calc(task='DV',
    #         dataset='iris')
    # sv_calc(task='FL', dataset='mnist')
    sv_calc(task='RI', dataset='wine')
