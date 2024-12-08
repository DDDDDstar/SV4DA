import sys
from collections import namedtuple

OUT_PRINT_FLUSH_INTERVAL = 5
UTILITY_RECORD_FILEWRITE_INTERVAL = 10
BENCHMARK = {
    'DV': {
        'iris': (120),
        'wine': (142),
    },
    'DSV': {
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


def para_set(args):
    required = ['task']
    # only required for bench tasks:
    bench_required = ['dataset']
    # only required for user-specific tasks:
    user_required = ['utility_function', 'player_num']
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
             'privacy_protection_level': 0.0,
             'utility_function': None
             }
    for required_key in required:
        if required_key not in args:
            print(f'ERROR: Missing required argument: {required_key}')
            return -1
    if args.get('task') in BENCHMARK:
        for bench_key in bench_required:
            if bench_key not in args:
                print(
                    f'ERROR: Missing required argument for benchmark task: {bench_key}')
                return -1
    else:
        for user_key in user_required:
            if user_key not in args:
                print(
                    f'ERROR: Missing required argument for user-specific task: {user_key}')
                return -1

    args.update({key: value for key, value in paras.items() if key not in args})
    return 0


def open_log_file(log_file):
    if log_file != 'std':
        try:
            file = open(log_file, 'w')
            sys.stdout = file
        except Exception as e:
            print(f"ERROR: Open log file error:\n{e}")
            return -1
    return 0


def config(args):
    if para_set(args) == -1:
        exit(-1)
    print(f'Experiment arguments:\n{args}')
    ARGS = namedtuple('ARGS', args.keys())
    args = ARGS(**args)

    if open_log_file(args.log_file) == -1:
        exit(-1)

    return args
