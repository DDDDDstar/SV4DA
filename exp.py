import argparse

from svbench import sv_calc


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--log_file', type=str, default='',
                        help="path of log file")
    parser.add_argument('--num_parallel_threads', type=int, default=1,
                        help="number of parallelThreads")
    parser.add_argument('--manual_seed', type=int, default=42,
                        help="random seed")

    # dataset parameters
    parser.add_argument('--dataset', type=str, default="mnist",
                        help="{MNIST, Iris}")

    # task parameters
    parser.add_argument('--task', type=str, default="DV",
                        help="{DV, FL, FA}")

    # SV parameters
    parser.add_argument('--conv_check_num', type=int, default=5,
                        help="SV cache_size")
    parser.add_argument('--base_algo', type=str, default="MC",
                        help="{MC, RE, MLE, GT, CP}")
    parser.add_argument('--convergence_threshold', type=float, default=0.05,
                        help="approximation convergence_threshold")

    parser.add_argument('--sampling_strategy', type=str, default="random",
                        help="{random, antithetic, stratified}")
    parser.add_argument('--optimization_strategy', type=bool, default=False,
                        help="")
    parser.add_argument('--TC_threshold', type=float, default=0.01,
                        help="truncation threshold")

    # SV's privacy protection parameters
    parser.add_argument('--privacy_protection_measure', type=str, default=None,
                        help="{None, DP, QT, DR}")
    parser.add_argument('--privacy_protection_level', type=float, default=0.0,
                        help="privacy_protection_level")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = args_parser()
    sv_calc(task=args.task,
            dataset=args.dataset,
            base_algo=args.base_algo,
            conv_check_num=args.conv_check_num,
            convergence_threshold=args.convergence_threshold,
            sampling_strategy=args.sampling_strategy,
            optimization_strategy=args.optimization_strategy,
            TC_threshold=args.TC_threshold,
            privacy_protection_measure=args.privacy_protection_measure,
            privacy_protection_level=args.privacy_protection_level,
            log_file=args.log_file,
            num_parallel_threads=args.num_parallel_threads,
            manual_seed=args.manual_seed
            )
