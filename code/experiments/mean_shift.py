import functools as funct

import numpy as np
import optuna
from clustering import *
from metrics import *
from optuna_sacred_adaptor import ConditionedTrial, run_sacred_grid_search, trial_wrapper, accuracy_from_contingency, TrialAbort
from pre_process import *
from serialisation import *

NAME = 'mean_shift'

def raw_pipeline(data_folder: str, eval_split: str, label_pipe_cache: SimpleCachedPipeline,
                 feature_pipe_cache: SimpleCachedPipeline, _config: Dict[str, Any], _seed: int,
                 cons_by_filter: Dict[str, Dict[str, Tuple[Pipeline, Pipeline]]],
                 include_save_operations: bool = True) -> MultiPipeline:
    label_pipe = PipelineAdaptorModule(selection_criterion=None,
                                       pipe_module=Sen1Floods11DataDingsDatasource(data_folder, type=TYPE_LABEL,
                                                                                   split=eval_split,
                                                                                   as_array=True),
                                       dataset_name='valid_labels'
                                       )
    label_pipe_cache.set_module(label_pipe)
    feature_space = _config['feature_space']
    train_cons, valid_cons = cons_by_filter[_config['filter']][feature_space]
    feature_pipe = PipelineAdaptorModule(selection_criterion=None,
                                         pipe_module=valid_cons,
                                         dataset_name='valid_features'
                                         )
    feature_pipe_cache.set_module(feature_pipe)
    data_construction = MultiDistributorModule([
        feature_pipe_cache,
        label_pipe_cache
    ])
    seq = [
        data_construction,
        SupervisedSklearnAdaptorModule(
            transformer=SupervisedSklearnAdaptor('clustering.MeanShiftWithRobustSeeding',
                                                 params={
                                                     'bandwidth': _config['bandwidth'],
                                                     'bin_seeding': _config['bin_seeding'],
                                                     'min_bin_freq': _config.get('min_bin_freq', 1),
                                                     'cluster_all': _config['cluster_all'],
                                                     # contrary to what the documentation states, this also allows
                                                     # the mode calculation to happen in parallel
                                                     # (parallelize all seeds)
                                                     # Furthermore this requires the following PR to be merged:
                                                     # https://github.com/scikit-learn/scikit-learn/pull/21845
                                                     # This is present in sklearn version 1.0.2+
                                                     'n_jobs': -1
                                                 },
                                                 per_data_point=True),
            feature_criterion=NameSelectionCriterion('valid_features'),
            label_criterion=NameSelectionCriterion('valid_labels'),
            prediction_dataset_name='valid_clustering',
            prediction_channel_name='valid_clustering'
        ),
        PipelineAdaptorModule(selection_criterion=NameSelectionCriterion('valid_clustering'),
                              pipe_module=PerImageClusteringModule(method='label_normalize',
                                                                   kwargs={'allow_region_props_calc': False}),
                              dataset_name='valid_clustering',
                              keep_source=False)
    ]
    if include_save_operations:
        seq.append(MetricsModule(prediction_criterion=NameSelectionCriterion('valid_clustering'),
                      label_criterion=NameSelectionCriterion('valid_labels'),
                      source_criterion=NameSelectionCriterion('valid_features'),
                      per_data_computation=ContingencyMatrixComputation('valid.contingency'),
                      delete_prediction=True))
    complete_pipe = MultiSequenceModule(seq, ignore_none=True)

    return complete_pipe
def pipeline(data_folder: str, cons_by_filter: Dict[str, Dict[str, Tuple[Pipeline, Pipeline]]],
             eval_split: str = SPLIT_VALIDATION) -> Callable:
    feature_pipe_cache = SimpleCachedPipeline()
    label_pipe_cache = SimpleCachedPipeline()

    @trial_wrapper
    def main(_config, _seed, _hook):
        complete_pipe = raw_pipeline(data_folder, eval_split, label_pipe_cache, feature_pipe_cache, _config, _seed, cons_by_filter)
        print(f'Serializing pipeline for config {_config}')
        with _hook.open_artifact_file('pipeline.json', 'w') as fd:
            serialize(fd, complete_pipe)
        print('Serialisation completed to file pipeline.json')
        summary = Summary([], lambda a: '')
        summary.set_hook(_hook.add_result_metric_prefix('valid.contingency'))
        print('Starting Pipeline')
        t = time.time()
        res_summary = complete_pipe(summary)
        t = time.time() - t
        print(f'Pipeline completed in {t:.3f}s')

        final_contingency_matrices = [d[0] for k, d in _hook.recorded_metrics.items() if k.startswith('valid')]
        print(f'Found {len(final_contingency_matrices)} matrices.')
        accuracy_values = [accuracy_from_contingency(d,True) for d in final_contingency_matrices]
        accuracy_values = [v for v in accuracy_values if v[1] > 0]
        print(f'Found {len(accuracy_values)} non-zero accuracy values.')
        res = sum(map(lambda t: t[0], accuracy_values)) / sum(map(lambda t: t[1], accuracy_values)) if len(
            accuracy_values) >= 1 else 0.0
        print(f'Mean Overall Accuracy is {res}.')
        return res
    return  main
def execute_grid_search_experiments(data_folder: str, experiment_base_folder: str,
                                    use_datadings: bool = True, seed: Optional[int] = None,
                                    timeout: Optional[int] = None,
                                    filters: Optional[List[str]] = None,
                                    redirect_output: bool = True):
    ex_name = NAME + '_grid_search'
    print('Constructing feature spaces.')
    cons_by_filter: Dict[str, Dict[str, Tuple[Any, Any]]] = {}
    all_filters = FILTER_METHODS.copy()
    all_filters.append(None)
    if filters is not None:
        flattened_aliases = set(funct.reduce(lambda l1, l2: l1 + l2, FILTER_ALIASES.values()))
        for filter_method in filters:
            if filter_method not in flattened_aliases:
                raise ValueError(f'Invalid filter {filter_method}. Expected one of {str(all_filters)}.')
        selected_filters = filters
    else:
        selected_filters = all_filters
    f_names = {}

    for filter_method in all_filters:
        train_cons = construct_clustering_feature_spaces(data_folder, split='train', in_memory=True,
                                              filter_method=filter_method, use_datadings=use_datadings,
                                              as_array=True)
        valid_cons = construct_clustering_feature_spaces(data_folder, split='valid', in_memory=True,
                                              filter_method=filter_method, use_datadings=use_datadings,
                                              as_array=True)
        cur_f_names = [t[0] for t in
                       train_cons]  # notice that we know that the filter does not influence the feature spaces
        print(f'Created {len(cur_f_names)} feature spaces [{", ".join(cur_f_names)}] for filter {filter_method}')
        cons_by_filter[filter_method] = {k1: (v1, v2) for (k1, v1), (k2, v2) in zip(train_cons, valid_cons)}
        f_names[filter_method] = cur_f_names
    print(f'Construction complete.')

    def config(trial: ConditionedTrial):
        filter = trial.suggest_categorical('filter', all_filters, selected_choices=selected_filters, condition=True)
        feature_space = trial.suggest_categorical('feature_space', f_names[filter], condition=True)
        bandwidth = trial.suggest_uniform('bandwidth', 1e-12, 1e3, selected_choices=[0.5, 0.25, 0.1, 0.075, 0.05, 0.025, 0.01])
        bin_seeding = trial.suggest_categorical('bin_seeding', [True, False], selected_choices=[True], condition=True)
        if bin_seeding:
            min_bin_freq = trial.suggest_int('min_bin_freq', 1, 1_000, selected_choices=[50])
        trial.leave_condition('bin_seeding')
        cluster_all = trial.suggest_categorical('cluster_all', [True, False], selected_choices=[True, False])

        # if per_data:
        #     k = trial.suggest_int('per_data_k', 1, 1_000,
        #                       selected_choices=list(range(2,11))+list(range(15,51,5))+list(range(100, 600, 100)))
        # else:
        trial.leave_condition(['feature_space', 'filter'])

    main = pipeline(data_folder, cons_by_filter)

    run_sacred_grid_search(experiment_base_folder, ex_name, main, config, seed, timeout=timeout,
                           redirect_output=redirect_output, direction=optuna.study.StudyDirection.MAXIMIZE)

def execute_final_experiments(data_folder: str, experiment_base_folder: str,
                                    use_datadings: bool = True, seed: Optional[int] = None,
                                    timeout: Optional[int] = None,
                                    filters: Optional[List[str]] = None,
                                    redirect_output: bool = True):
    ex_name = NAME + '_final'
    for split in [SPLIT_TEST, SPLIT_BOLIVIA]:
        print('Constructing feature spaces.')
        cons_by_filter, f_names, all_filters, selected_filters = default_clustering_feature_space_construction(data_folder, filters, use_datadings, split)
        print(f'Construction complete.')

        def config(trial: ConditionedTrial):
            filter = trial.suggest_categorical('filter', all_filters, selected_choices=selected_filters, condition=True)
            tt = trial.suggest_categorical('tt', [0, 1, 2, 3])
            if tt == 0:
                return {'feature_space': 'SAR',
                        'bandwidth': 0.5,
                        'bin_seeding': True,
                        'min_bin_freq': 50,
                        'cluster_all': False}
            elif tt == 1:
                return {'feature_space': 'cAWEI',
                        'bandwidth': 0.5,
                        'bin_seeding': True,
                        'min_bin_freq': 50,
                        'cluster_all': False}
            elif tt == 2:
                return {'feature_space': 'cAWEI+NDVI',
                        'bandwidth': 0.5,
                        'bin_seeding': True,
                        'min_bin_freq': 50,
                        'cluster_all': False}
            elif tt == 3:
                return {'feature_space': 'SAR_cNDWI',
                        'bandwidth': 0.5,
                        'bin_seeding': True,
                        'min_bin_freq': 50,
                        'cluster_all': False}
            trial.leave_condition(['tt', 'filter'])

        try:
            run_sacred_grid_search(experiment_base_folder, ex_name+'_'+split,
                                   pipeline(data_folder, cons_by_filter, eval_split=split),
                                   config, seed, timeout=timeout,
                                   redirect_output=redirect_output, direction=optuna.study.StudyDirection.MAXIMIZE)
        except TrialAbort:
            print('Execution was aborted. Trying next experiment!', file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)