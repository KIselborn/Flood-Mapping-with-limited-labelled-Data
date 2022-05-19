import functools as funct

import numpy as np
import optuna
from clustering import *
from metrics import *
from optuna_sacred_adaptor import ConditionedTrial, run_sacred_grid_search, trial_wrapper, accuracy_from_contingency, TrialAbort
from pre_process import *
from serialisation import *

NAME = 'segmentation_clustering_2'

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
    cluster_alg = _config['cluster_alg']
    if cluster_alg == 'Gaussian Mixture':
        adaptor = SupervisedSklearnAdaptor('sklearn.mixture.GaussianMixture',
                                           params={
                                               'n_components': _config['n_components'],
                                               'covariance_type': _config['cov_type'],
                                               'init_params': _config['init_params'],
                                               'tol': _config['tol'],
                                               'random_state': _seed
                                           },
                                           per_data_point=True,
                                           save_file=None,
                                           clear_on_predict=False)
        module = SupervisedSklearnAdaptorModule(adaptor,
                                                feature_criterion=NameSelectionCriterion('valid_features'),
                                                label_criterion=NameSelectionCriterion('valid_labels'),
                                                prediction_dataset_name='valid_clustering',
                                                prediction_channel_name='valid_clustering')
    elif cluster_alg == 'Quickshift':
        qm = PerImageClusteringModule(method='quickshift',
                                      out_channel='valid_clustering',
                                      kwargs={
                                        'random_seed': _seed,
                                        'ratio': _config['quickshift_ratio'],
                                        'kernel_size': (_config['quickshift_kernel_size'] - 1) / 6.0,
                                        'max_dist': _config['quickshift_max_dist'],
                                        'sigma': _config['quickshift_sigma'],
                                        'convert2lab': _config[feature_space+'_convert2lab']
                                      })
        module = PipelineAdaptorModule(selection_criterion=NameSelectionCriterion('valid_features'),
                                       pipe_module=qm,
                                       dataset_name='valid_clustering',
                                       keep_source=True)
    # elif 'BIRCH' in cluster_alg:
    #     threshold = _config['threshold']
    #     branching_factor = _config['branching_factor']
    #     if 'DB-Scan' in cluster_alg:
    #         pretext_alg = SerializableDBScan(eps=_config['b_eps'], min_samples=_config['b_min_samples'],
    #                                          metric=_config['b_dbscan_metric'])
    #     # elif 'Hierarchical' in cluster_alg:
    #     #     pretext_alg = SerializableAgglomerative(_config['hierarchical_clusters'], affinity=_config['affinity'],
    #     #                                             linkage=_config['linkage'])
    #     elif 'Direct' in cluster_alg:
    #         pretext_alg = None
    #     else:
    #         raise ValueError('Unknown clustering algorithm ' + cluster_alg)
    #     module = SupervisedSklearnAdaptorModule(
    #         transformer=SupervisedSklearnAdaptor('sklearn.cluster.Birch',
    #                                              params={
    #                                                 'threshold': threshold,
    #                                                 'branching_factor': branching_factor,
    #                                                 'n_clusters': pretext_alg
    #                                              },
    #                                              per_data_point=True),
    #                                              feature_criterion=NameSelectionCriterion('valid_features'),
    #                                              label_criterion=NameSelectionCriterion('valid_labels'),
    #                                              prediction_dataset_name='valid_prediction',
    #                                              prediction_channel_name='valid_prediction'
    #     )
    else:
        raise ValueError('Unknown clustering algorithm ' + cluster_alg)
    seq = [
        data_construction,
        module,
        # PipelineAdaptorModule(selection_criterion=NameSelectionCriterion('valid_prediction'),
        #                       pipe_module=PerImageClusteringModule(method='label_normalize',
        #                                                            kwargs={'allow_region_props_calc': False}),
        #                       dataset_name='valid_prediction',
        #                       keep_source=False),
        # evaluate if lazy
        PipelineAdaptorModule(NameSelectionCriterion(name='valid_clustering'),
                              ShapelessInMemoryModule())
    ]
    if include_save_operations:
        seq.append(MetricsModule(prediction_criterion=NameSelectionCriterion('valid_clustering'),
                                label_criterion=NameSelectionCriterion('valid_labels'),
                                source_criterion=None,
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
        complete_pipe = raw_pipeline(data_folder, eval_split, label_pipe_cache, feature_pipe_cache, _config,
                                     _seed, cons_by_filter)
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
        accuracy_values = [accuracy_from_contingency(d, True) for d in final_contingency_matrices]
        accuracy_values = [v for v in accuracy_values if v[1] > 0]
        print(f'Found {len(accuracy_values)} non-zero accuracy values.')
        res = sum(map(lambda t: t[0], accuracy_values)) / sum(map(lambda t: t[1], accuracy_values)) if len(
            accuracy_values) >= 1 else 0.0
        print(f'Mean Overall Accuracy is {res}.')
        return res

    return main

def execute_grid_search_experiments(data_folder: str, experiment_base_folder: str,
                                    use_datadings: bool = True, seed: Optional[int] = None,
                                    timeout: Optional[int] = None,
                                    filters: Optional[List[str]] = None,
                                    redirect_output: bool = True):
    ex_name = NAME + '_grid_search'
    print('Constructing feature spaces.')
    cons_by_filter, f_names, all_filters, selected_filters = default_clustering_feature_space_construction(data_folder, filters, use_datadings, SPLIT_VALIDATION)
    print(f'Construction complete.')

    def config(trial: ConditionedTrial):
        filter = trial.suggest_categorical('filter', all_filters, selected_choices=selected_filters, condition=True)
        feature_space = trial.suggest_categorical('feature_space',f_names[filter], condition=True)
        cluster_alg = trial.suggest_categorical('cluster_alg', ['Gaussian Mixture', 'Quickshift'],#
                                                condition=True)
        if cluster_alg == 'Gaussian Mixture': # 24 possibilities per feature space => 552 estimated at 15min
            cov_type = trial.suggest_categorical('cov_type', ['full', 'tied', 'diag', 'spherical'],
                                                 selected_choices=['full', 'tied', 'diag'])
            init_params = trial.suggest_categorical('init_params', ['kmeans', 'random'])
            tol = trial.suggest_uniform('tol', 1e-12, 1.0, selected_choices=[1e-3])
            n_components = trial.suggest_int('n_components', 2, 10_000, selected_choices=[2, 5, 10, 15])
        elif cluster_alg == 'Quickshift': # 40 possibilities per feature space => 920 executions estimated at 2min
            trial.suggest_categorical(feature_space+'_convert2lab', [True, False], selected_choices=list({feature_space == 'RGB', False}))
            trial.suggest_uniform('quickshift_ratio', 0.0, 1.0, selected_choices=[1.0, 0.875, 0.75, 0.625, 0.5])
            trial.suggest_int('quickshift_kernel_size', 7, 37, selected_choices=[7, 13])
            trial.suggest_uniform('quickshift_max_dist', 1e-12, 10.0, selected_choices=[2.0, 4.0, 6.0, 8.0])
            trial.suggest_uniform('quickshift_sigma', 0.0, 5.0, selected_choices=[0.0])
        else:
            raise ValueError('Unknown clustering algorithm '+cluster_alg)

        # if per_data:
        #     k = trial.suggest_int('per_data_k', 1, 1_000,
        #                       selected_choices=list(range(2,11))+list(range(15,51,5))+list(range(100, 600, 100)))
        # else:
        trial.leave_condition(['feature_space', 'filter'])

    main = pipeline(data_folder, cons_by_filter, SPLIT_VALIDATION)

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
            cluster_alg = trial.suggest_categorical('cluster_alg', ['Gaussian Mixture', 'Quickshift'],#
                                                    condition=True)
            if cluster_alg == 'Gaussian Mixture':
                tol = trial.suggest_uniform('tol', 1e-12, 1.0, selected_choices=[1e-3])
                feature_space = trial.suggest_categorical(cluster_alg+'feature_space', ['SAR_cAWEI', 'SAR_O3', 'cNDWI', 'O3', 'SAR'], condition=True)
                if feature_space == 'SAR_cAWEI':
                    cov_type = trial.suggest_categorical(feature_space+'cov_type', ['full', 'tied', 'diag', 'spherical'],
                                                         selected_choices=['diag'])
                    init_params = trial.suggest_categorical(feature_space+'init_params', ['kmeans','random'], selected_choices=['random'])
                    n_components = trial.suggest_int(feature_space+'n_components', 2, 10_000, selected_choices=[2])
                elif feature_space == 'SAR_O3':
                    cov_type = trial.suggest_categorical(feature_space+'cov_type', ['full', 'tied', 'diag', 'spherical'],
                                                         selected_choices=['diag'])
                    init_params = trial.suggest_categorical(feature_space+'init_params', ['kmeans','random'], selected_choices=['random'])
                    n_components = trial.suggest_int(feature_space+'n_components', 2, 10_000, selected_choices=[10])
                elif feature_space == 'cNDWI':
                    cov_type = trial.suggest_categorical(feature_space+'cov_type', ['full', 'tied', 'diag', 'spherical'],
                                                         selected_choices=['tied'])
                    init_params = trial.suggest_categorical(feature_space+'init_params', ['kmeans','random'], selected_choices=['kmeans'])
                    n_components = trial.suggest_int(feature_space+'n_components', 2, 10_000, selected_choices=[2])
                elif feature_space == 'O3':
                    cov_type = trial.suggest_categorical(feature_space+'cov_type', ['full', 'tied', 'diag', 'spherical'],
                                                         selected_choices=['tied'])
                    init_params = trial.suggest_categorical(feature_space+'init_params', ['kmeans','random'], selected_choices=['kmeans'])
                    n_components = trial.suggest_int(feature_space+'n_components', 2, 10_000, selected_choices=[10])
                elif feature_space == 'SAR':
                    cov_type = trial.suggest_categorical(feature_space+'cov_type', ['full', 'tied', 'diag', 'spherical'],
                                                         selected_choices=['diag'])
                    init_params = trial.suggest_categorical(feature_space+'init_params', ['kmeans','random'], selected_choices=['random'])
                    n_components = trial.suggest_int(feature_space+'n_components', 2, 10_000, selected_choices=[2])
                else:
                    raise RuntimeError()
                trial.leave_condition(['feature_space', 'filter', 'cluster_alg'])
                return {'n_components':n_components, 'init_params':init_params, 'cov_type': cov_type, 'feature_space':feature_space}
            elif cluster_alg == 'Quickshift':
                feature_space = trial.suggest_categorical(cluster_alg+'feature_space', ['SAR_cAWEI+NDVI','cAWEI+NDVI','SAR'], condition=True)
                trial.suggest_categorical(feature_space+'_convert2lab', [True, False], selected_choices=[False])
                trial.suggest_int('quickshift_kernel_size', 7, 37, selected_choices=[13])
                trial.suggest_uniform('quickshift_max_dist', 1e-12, 10.0, selected_choices=[8.0])
                trial.suggest_uniform('quickshift_sigma', 0.0, 5.0, selected_choices=[0.0])
                if feature_space == 'cAWEI+NDVI':
                    quickshift_ratio = trial.suggest_uniform(feature_space+'quickshift_ratio', 0.0, 1.0, selected_choices=[1.0])
                else:
                    quickshift_ratio = trial.suggest_uniform(feature_space+'quickshift_ratio', 0.0, 1.0, selected_choices=[0.875])
                trial.leave_condition(['feature_space', 'filter', 'cluster_alg'])

                return {'quickshift_ratio': quickshift_ratio, 'feature_space': feature_space}
            else:
                raise ValueError('Unknown clustering algorithm '+cluster_alg)

            # if per_data:
            #     k = trial.suggest_int('per_data_k', 1, 1_000,
            #                       selected_choices=list(range(2,11))+list(range(15,51,5))+list(range(100, 600, 100)))
            # else:

        try:
            run_sacred_grid_search(experiment_base_folder, ex_name+'_'+split,
                                   pipeline(data_folder, cons_by_filter, eval_split=split),
                                   config, seed, timeout=timeout,
                                   redirect_output=redirect_output, direction=optuna.study.StudyDirection.MAXIMIZE)
        except TrialAbort:
            print('Execution was aborted. Trying next experiment!', file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
