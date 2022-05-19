import argparse
import os

import torch

import experiments.segmentation_clustering as ex

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # make it work deterministically on cuda 10.2+
torch.use_deterministic_algorithms(True)

DEF_TIMEOUT = None  # (2*24*60*60+23*60*60)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_folder')
    parser.add_argument('experiment_folder')
    parser.add_argument('--datadings', dest='use_datadings', action='store_true')
    parser.add_argument('--no-datadings', dest='use_datadings', action='store_false')
    parser.add_argument('--redirect-console', dest='redirect_output', action='store_true')
    parser.add_argument('--no-redirect-console', dest='redirect_output', action='store_false')
    parser.set_defaults(use_datadings=True)
    parser.set_defaults(redirect_output=True)
    parser.add_argument('-s', '--seed', dest='seed', default=None, type=int, help='Seed to use for reproducability.')
    parser.add_argument('-ss', '--seeds', nargs='+', dest='seeds', default=None, help='Set of seeds to use for reproducability.')
    parser.add_argument('-t', '--timeout', dest='timeout', default=DEF_TIMEOUT, type=int, help='Optuna timeout in s')
    #parser.add_argument('-j', '--n_jobs', dest='n_jobs', default=-1, type=int, help='Number of concurrent jobs to run, if supported by the pipeline')
    parser.add_argument('--filters', nargs='+', dest='filters', default=None, help='Filters to use for the experiments')
    parser.add_argument('--additional_list', nargs='+', dest='additional_list', default=None, help='For the combined experiment this represents the tested combined algorithms. Set to None to test all')
    args = parser.parse_args()
    timeout = args.timeout
    if timeout is not None and timeout < 0:
        raise ValueError('Negative Timeout given!')
    if args.seed is not None and args.seeds is not None:
        raise RuntimeError('Cannot specify both a set of seeds to test and a single seed to test!!!')
    elif args.seed is not None:
        seeds = [args.seed]
    elif args.seeds is not None:
        seeds = [int(seed) for seed in args.seeds]
    else:
        print('No seed specified. Running without seed.')
        seeds = [None]
    for seed in seeds:
        if args.additional_list is None:
            experiment_instances = ex.execute_final_experiments(args.data_folder, args.experiment_folder,
                                                                      args.use_datadings, seed, timeout, args.filters,
                                                                      args.redirect_output)
        else:
            experiment_instances = ex.execute_final_experiments(args.data_folder, args.experiment_folder,
                                                                      args.use_datadings, seed, timeout, args.filters,
                                                                      args.redirect_output, args.additional_list)
