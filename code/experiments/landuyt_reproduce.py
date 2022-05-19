import functools as funct

import optuna
from clustering import *
from metrics import *
from optuna_sacred_adaptor import ConditionedTrial, run_sacred_grid_search, trial_wrapper, accuracy_from_contingency, \
    TrialAbort
from pre_process import *
from serialisation import *

NAME = 'landuyt_reproduce'


def execute_grid_search_experiments_old(data_folder: str, experiment_base_folder: str,
                                        use_datadings: bool = True, seed: Optional[int] = None,
                                        timeout: Optional[int] = None,
                                        filters: Optional[List[str]] = None):
    ex_name = NAME + '_grid_search'
    print('Constructing feature spaces.')
    cons_by_filter = {}
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
        data_cons = construct_feature_spaces(data_folder, split='valid', in_memory=True,
                                             filter_method=filter_method, use_datadings=use_datadings,
                                             add_optical=True, allow_lin_ratio= True)
        cur_f_names = [t[0] for t in
                       data_cons]  # notice that we know that the filter does not influence the feature spaces
        print(f'Created feature spaces [{", ".join(cur_f_names)}] for filter {filter_method}')
        cons_by_filter[filter_method] = dict(data_cons)
        f_names[filter_method] = cur_f_names
    print(f'Construction complete.')

    def config(trial: ConditionedTrial):
        selected_filter = trial.suggest_categorical('selected_filter', all_filters, selected_choices=selected_filters,
                                                    condition=True)
        fs = trial.suggest_categorical('feature_space', f_names[selected_filter],
                                       selected_choices=['SAR', 'SAR+R', 'SAR_O3', 'SAR_OPT', 'SAR+R_O3', 'SAR+R_OPT'],
                                       condition=True)
        available_channels = []
        if 'SAR' in fs:
            available_channels.append(['VV', 'VH'])
            if 'SAR+R' in fs:
                available_channels.append(['VV', 'VH', 'VV-VH lin. Ratio'])
        if 'NDWI' in fs:
            available_channels.append(['NDWI', 'MNDWI1'])
            if 'NDVI' in fs:
                available_channels.append(['NDWI', 'MNDWI1', 'NDVI'])
        if 'AWEI' in fs:
            available_channels.append(['AWEI', 'AWEISH'])
            if 'NDVI' in fs:
                available_channels.append(['AWEI', 'AWEISH', 'NDVI'])
            if 'NDWI' in fs:
                available_channels.append(['AWEI', 'AWEISH', 'NDWI', 'MNDWI1'])
                if 'NDVI' in fs:
                    available_channels.append(['AWEI', 'AWEISH', 'NDWI', 'MNDWI1', 'NDVI'])
        trial.suggest_categorical('quickshift_channels', available_channels,
                                  selected_choices=[['VV', 'VH']])
        trial.leave_condition('feature_space')
        trial.leave_condition('selected_filter')

        trial.suggest_uniform('quickshift_ratio', 0.0, 1.0, selected_choices=[1.0])
        trial.suggest_int('quickshift_kernel_size', 7, 37, selected_choices=[7])
        trial.suggest_uniform('quickshift_max_dist', 1.0, 10.0, selected_choices=[4.0])
        trial.suggest_uniform('quickshift_sigma', 0.0, 5.0, selected_choices=[0.0])

        trial.suggest_int('post_quickshift_connectivity', 1, 2, selected_choices=[1])

        trial.suggest_categorical('k_means_stats', [['mean_intensity']])  # , ['mean_intensity', 'standard_deviation']])
        trial.suggest_int('k_means_clusters', 2, 100, selected_choices=[i for i in range(2, 16)])

        trial.suggest_int('clustering_refinement_connectivity', 1, 2, selected_choices=[1])
        trial.suggest_int('clustering_refinement_convergence_threshold', 1, 100, selected_choices=[10])
        trial.suggest_uniform('clustering_refinement_perim_area_threshold', 3.0, 48.0,
                              selected_choices=[8.0, 12.0, 16.0])
        trial.suggest_uniform('clustering_refinement_rel_std_threshold', 0.25, 4.0, selected_choices=[1.0])
        trial.suggest_int('clustering_refinement_max_iter', 0, 20, selected_choices=[10])
        trial.suggest_categorical('clustering_refinement_remove_singular',
                                  choices=[True, False],
                                  selected_choices=[True])
        greedy = trial.suggest_categorical('clustering_refinement_greedy_strategy',
                                           choices=[True, False],
                                           selected_choices=[True],
                                           condition=True)
        if not greedy:
            trial.suggest_int('clustering_refinement_max_merge_trials', 1, 1000, selected_choices=[100])
        trial.leave_condition('clustering_refinement_greedy_strategy')

        trial.suggest_categorical('threshold_alg', ['otsu', 'ki'])
        trial.suggest_categorical('bin_count', ['tile_dim', 'auto'])
        use_tiled = trial.suggest_categorical('threshold_use_tiled', [True], condition=True)  # [True, False]
        if use_tiled:
            trial.suggest_categorical('threshold_tile_dim', [32, 64, 128, 256], selected_choices=[256])
            trial.suggest_categorical('threshold_force_merge_on_failure', [True, False], selected_choices=[False])
            trial.suggest_float('threshold_percentile', 0.8, 0.99, selected_choices=[0.95])
        trial.leave_condition('threshold_use_tiled')

    @trial_wrapper
    def main(_config, _seed, _hook):
        print('--------------------------------------------------------------------------')
        filter_method = _config['selected_filter']
        name = _config['feature_space']
        print('Constructing Pipe for', name, 'filtered by', filter_method)
        threshold_args = {
            'print_warning': False,
            'bin_count': _config['bin_count'],
            'use_tiled': _config['threshold_use_tiled']
        }
        if _config['threshold_use_tiled']:
            threshold_args['tile_dim'] = _config['threshold_tile_dim']
            threshold_args['force_merge_on_failure'] = _config['threshold_force_merge_on_failure']
            threshold_args['percentile'] = _config['threshold_percentile']
        threshold_alg = 'classification.KIPerImageThresholdingClassifier'
        if _config['threshold_alg'] == 'otsu':
            threshold_alg = 'classification.OtsuPerImageThresholdingClassifier'
        thresholding = UnsupervisedSklearnAdaptor(threshold_alg,
                                                  params=threshold_args,
                                                  per_channel=True,
                                                  per_data_point=True,
                                                  image_features=True,
                                                  save_file='thresholds.pkl')
        bitwise_combine = UnsupervisedSklearnAdaptor('classification.BitwiseAndChannelCombiner',
                                                     per_data_point=True,
                                                     allow_no_fit=True)
        quickshift_args = {
            'random_seed': _seed,
            'ratio': _config['quickshift_ratio'],
            'kernel_size': (_config['quickshift_kernel_size'] - 1) / 6.0,
            'max_dist': _config['quickshift_max_dist'],
            'sigma': _config['quickshift_sigma']
        }
        connectivity_args = {
            'connectivity': _config['post_quickshift_connectivity']
        }
        k_means_args = {
            'n_clusters': _config['k_means_clusters'],
            'random_state': _seed
        }
        k_means = UnsupervisedSklearnAdaptor('sklearn.cluster.KMeans',
                                             per_data_point=True,
                                             params=k_means_args,
                                             save_file='k_means.pkl')
        clustering_refinement_args = {
            'connectivity': _config['clustering_refinement_connectivity'],
            'convergence_threshold': _config['clustering_refinement_convergence_threshold'],
            'perim_area_threshold': _config['clustering_refinement_perim_area_threshold'],
            'rel_std_threshold': _config['clustering_refinement_rel_std_threshold'],
            'max_iter': _config['clustering_refinement_max_iter'],
            'greedy_merge_strategy': _config['clustering_refinement_greedy_strategy'],
            'remove_singular': _config['clustering_refinement_remove_singular'],
            'random_state': _seed
        }
        if not _config['clustering_refinement_greedy_strategy']:
            clustering_refinement_args['max_merge_trials'] = _config['clustering_refinement_max_merge_trials']
        complete_pipe = MultiSequenceModule([
            # {} -> {'features'}
            PipelineAdaptorModule(selection_criterion=None,
                                  pipe_module=SequenceModule([
                                      cons_by_filter[filter_method][name],
                                      DistributorModule([
                                          SequenceModule([
                                              WhitelistModule(['VV', 'VH']),
                                              UnsupervisedSklearnAdaptorModule(thresholding,
                                                                               do_predict=False),
                                              TerminationModule()
                                          ], ignore_none=True)
                                      ], keep_source=True, ignore_none=True)
                                  ]),
                                  dataset_name='features'
                                  ),
            # {'features'} -> {'features', 'labels'}
            PipelineAdaptorModule(selection_criterion=None,
                                  pipe_module=Sen1Floods11DataDingsDatasource(data_folder, type=TYPE_LABEL,
                                                                              split=SPLIT_VALIDATION),
                                  dataset_name='labels'
                                  ),
            # {'features', 'labels'} -> {'features', 'labels', 'clustering'}
            PipelineAdaptorModule(NameSelectionCriterion(name='features'),
                                  SequenceModule([WhitelistModule(_config['quickshift_channels']),
                                                  PerImageClusteringModule(method='quickshift',
                                                                           out_channel='clustering',
                                                                           kwargs=quickshift_args),
                                                  PerImageClusteringModule(method='connectivity',
                                                                           out_channel='clustering',
                                                                           kwargs=connectivity_args),
                                                  ShapelessInMemoryModule()]),
                                  keep_source=True,
                                  dataset_name='clustering'
                                  ),
            MetricsModule(prediction_criterion=NameSelectionCriterion('clustering'),
                          label_criterion=NameSelectionCriterion('labels'),
                          source_criterion=NameSelectionCriterion('features'),
                          per_data_computation=SequenceMetricComputation([
                              ContingencyMatrixComputation('quickshift.contingency'),
                              ClusterSpatialDistributionComputation('quickshift.spatial_dist')
                          ]),
                          delete_label=False,
                          delete_prediction=False,
                          delete_source=False),
            # {'features', 'labels', 'clustering'} - > {'features', 'labels', 'clustering'}
            PerImageClusteringRefinementModule(label_criterion=NameSelectionCriterion(name='clustering'),
                                               intensity_criterion=NameSelectionCriterion(name='features'),
                                               **clustering_refinement_args),
            # {'features', 'labels', 'clustering'} - > {'features', 'labels', 'clustering'}
            PipelineAdaptorModule(NameSelectionCriterion(name='clustering'),
                                  PerImageClusteringModule(method='label_normalize')),
            # {'features', 'labels', 'clustering'} - > {'features', 'labels', 'clustering'}
            PipelineAdaptorModule(NameSelectionCriterion(name='clustering'),
                                  ShapelessInMemoryModule()),
            MetricsModule(prediction_criterion=NameSelectionCriterion('clustering'),
                          label_criterion=NameSelectionCriterion('labels'),
                          source_criterion=NameSelectionCriterion('features'),
                          per_data_computation=SequenceMetricComputation([
                              ContingencyMatrixComputation('refined.contingency'),
                              ClusterSpatialDistributionComputation('refined.spatial_dist')
                          ]),
                          delete_label=False,
                          delete_prediction=False,
                          delete_source=False),
            # {'features', 'labels', 'clustering'} -> {'features', 'labels', 'clustering', 'zonal_stats'}
            PerImageZonalStatsExtractorModule(label_criterion=NameSelectionCriterion(name='clustering'),
                                              intensity_criterion=NameSelectionCriterion(name='features'),
                                              intensity_dataset_result_name='zonal_stats',
                                              stats_of_interest=_config['k_means_stats'],
                                              keep_intensity=True,
                                              keep_label=True,
                                              no_stat_suffix=True),
            PipelineAdaptorModule(NameSelectionCriterion(name='zonal_stats'),
                                  ShapelessInMemoryModule()),
            # {'features', 'labels', 'clustering', 'zonal_stats'} ->
            # {'features', 'labels', 'clustering', 'zonal_stats', 'zonal_clustering'}
            PipelineAdaptorModule(NameSelectionCriterion(name='zonal_stats'),
                                  UnsupervisedSklearnAdaptorModule(k_means),  # do K-Means
                                  keep_source=True,
                                  dataset_name='zonal_clustering'),
            # {'features', 'labels', 'clustering', 'zonal_stats', 'zonal_clustering'} ->
            # {'features', 'labels', 'clustering', 'zonal_stats', 'zonal_clustering', 'zonal_clustering_result'}
            AssemblerModule(label_criterion=NameSelectionCriterion(name='clustering'),
                            value_criterion=NameSelectionCriterion(name='zonal_clustering'),
                            res_dataset_name='zonal_clustering_result',
                            delete_value=False,
                            delete_label=False),
            # {'features', 'labels', 'clustering', 'zonal_stats', 'zonal_clustering', 'zonal_clustering_result'} ->
            # {'labels', 'clustering', 'zonal_stats', 'zonal_clustering'}
            MetricsModule(prediction_criterion=NameSelectionCriterion('zonal_clustering_result'),
                          label_criterion=NameSelectionCriterion('labels'),
                          source_criterion=NameSelectionCriterion('features'),
                          per_data_computation=SequenceMetricComputation([
                              ContingencyMatrixComputation('k_means.contingency'),
                              ClusterSpatialDistributionComputation('kmeans.spatial_dist')
                          ]),
                          delete_label=False,
                          delete_prediction=True,
                          delete_source=True),

            # {'labels', 'clustering', 'zonal_stats', 'zonal_clustering'} ->
            # {'labels', 'clustering', 'zonal_stats', 'zonal_clustering'}
            PipelineAdaptorModule(NameSelectionCriterion(name='zonal_stats'),
                                  WhitelistModule(['VV', 'VH'])),
            # {'labels', 'clustering', 'zonal_stats', 'zonal_clustering'} ->
            # {'labels', 'clustering', 'zonal_clustering', 'zonal_clustering_stats'}
            PerDataPointStatsExtractorModule(
                label_criterion=NameSelectionCriterion(name='zonal_clustering'),
                intensity_criterion=NameSelectionCriterion(name='zonal_stats'),
                stats_of_interest=['mean_intensity'],
                intensity_dataset_result_name='zonal_clustering_stats',
                keep_label=True,
                no_stat_suffix=True),
            # {'labels', 'clustering', 'zonal_clustering', 'zonal_clustering_stats'}
            # -> {'labels', 'clustering', 'zonal_clustering', 'classification'}
            PipelineAdaptorModule(NameSelectionCriterion(name='zonal_clustering_stats'),
                                  SequenceModule([
                                      UnsupervisedSklearnAdaptorModule(thresholding, do_fit=False),
                                      UnsupervisedSklearnAdaptorModule(bitwise_combine, do_fit=False)
                                  ]),
                                  dataset_name='classification'),
            # {'labels', 'clustering', 'zonal_clustering', 'classification'} ->
            # {'labels', 'clustering', 'zonal_clustering_classification'}
            AssemblerModule(label_criterion=NameSelectionCriterion(name='zonal_clustering'),
                            value_criterion=NameSelectionCriterion(name='classification'),
                            res_dataset_name='zonal_clustering_classification'),
            # {'labels', 'clustering', 'zonal_clustering_classification'} ->
            # {'labels', 'clustering_classification'}
            AssemblerModule(label_criterion=NameSelectionCriterion(name='clustering'),
                            value_criterion=NameSelectionCriterion(name='zonal_clustering_classification'),
                            res_dataset_name='clustering_classification'),
            MetricsModule(prediction_criterion=NameSelectionCriterion('clustering_classification'),
                          label_criterion=NameSelectionCriterion('labels'),
                          per_data_computation=ContingencyMatrixComputation('final.contingency'),
                          delete_label=True,
                          delete_prediction=True)
        ])
        print('Serializing pipeline')
        with _hook.open_artifact_file('pipeline.json', 'w') as fd:
            serialize(fd, complete_pipe)
        print('Serialisation completed to file pipeline.json')
        summary = Summary([], lambda a: '')
        summary.set_hook(_hook.add_result_metric_prefix('final.contingency'))
        print('Starting Pipeline')
        t = time.time()
        res_summary = complete_pipe(summary)
        t = time.time() - t
        print(f'Pipeline completed in {t:.3f}s')

        def accuracy_from_contingency(d: dict) -> Tuple[float, int]:
            matrix = d[KEY_CONTINGENCY_MATRIX]
            ul = d[KEY_CONTINGENCY_UNIQUE_LABELS]
            up = d[KEY_CONTINGENCY_UNIQUE_PREDICTIONS]
            divisor = matrix[ul != -1].sum()
            dividend = (matrix[ul == 0, up == 0].sum() + matrix[ul == 1, up == 1].sum())
            return (0.0, 0) if divisor == 0 else (dividend / divisor, 1)

        final_contingency_matrices = [d[0] for k, d in _hook.recorded_metrics.items() if
                                      k.startswith('final.contingency')]
        accuracy_values = [accuracy_from_contingency(d) for d in final_contingency_matrices]
        res = sum(map(lambda t: t[0], accuracy_values)) / sum(map(lambda t: t[1], accuracy_values))
        print(f'Mean Overall Accuracy is {res}.')
        return res

    run_sacred_grid_search(experiment_base_folder, ex_name, main, config, seed, timeout=timeout)

def raw_pipeline(data_folder: str, eval_split: str, label_pipe_cache: SimpleCachedPipeline,
                  _config: Dict[str, Any], _seed: int,
                 cons_by_filter: Dict[str, Dict[str, Tuple[Pipeline, Pipeline]]],
                 include_save_operations: bool = True) -> MultiPipeline:
    filter_method = _config['selected_filter']
    name = _config['feature_space']
    print('Constructing Pipe for', name, 'filtered by', filter_method)
    threshold_args = {
        'print_warning': False,
        'bin_count': _config['bin_count'],
        'use_tiled': _config['threshold_use_tiled']
    }
    if _config['threshold_use_tiled']:
        threshold_args['tile_dim'] = _config['threshold_tile_dim']
        threshold_args['force_merge_on_failure'] = _config['threshold_force_merge_on_failure']
        threshold_args['percentile'] = _config['threshold_percentile']
    threshold_alg = 'classification.KIPerImageThresholdingClassifier'
    if _config['threshold_alg'] == 'otsu':
        threshold_alg = 'classification.OtsuPerImageThresholdingClassifier'
    thresholding = UnsupervisedSklearnAdaptor(threshold_alg,
                                              params=threshold_args,
                                              per_channel=True,
                                              per_data_point=True,
                                              image_features=True,
                                              save_file='thresholds.pkl')
    bitwise_combine = UnsupervisedSklearnAdaptor('classification.BitwiseAndChannelCombiner',
                                                 params={'reduction_axis': 0},
                                                 per_data_point=True,
                                                 allow_no_fit=True)
    quickshift_args = {
        'random_seed': _seed,
        'ratio': _config['quickshift_ratio'],
        'kernel_size': (_config['quickshift_kernel_size'] - 1) / 6.0,
        'max_dist': _config['quickshift_max_dist'],
        'sigma': _config['quickshift_sigma']
    }
    connectivity_args = {
        'connectivity': _config['post_quickshift_connectivity']
    }
    k_means_args = {
        'n_clusters': _config['k_means_clusters'],
        'random_state': _seed
    }
    k_means = UnsupervisedSklearnAdaptor('sklearn.cluster.KMeans',
                                         per_data_point=True,
                                         params=k_means_args,
                                         save_file='k_means.pkl')
    clustering_refinement_args = {
        'connectivity': _config['clustering_refinement_connectivity'],
        'convergence_threshold': _config['clustering_refinement_convergence_threshold'],
        'perim_area_threshold': _config['clustering_refinement_perim_area_threshold'],
        'rel_std_threshold': _config['clustering_refinement_rel_std_threshold'],
        'max_iter': _config['clustering_refinement_max_iter'],
        'greedy_merge_strategy': _config['clustering_refinement_greedy_strategy'],
        'remove_singular': _config['clustering_refinement_remove_singular'],
        'random_state': _seed
    }
    if not _config['clustering_refinement_greedy_strategy']:
        clustering_refinement_args['max_merge_trials'] = _config['clustering_refinement_max_merge_trials']

    _, valid_cons = cons_by_filter[_config['selected_filter']][_config['feature_space']]
    label_pipe_cache.set_module(MultiDistributorModule([PipelineAdaptorModule(selection_criterion=None,
                                                                        pipe_module=valid_cons,
                                                                        dataset_name='features'),
                                                  PipelineAdaptorModule(selection_criterion=None,
                                                                        pipe_module=Sen1Floods11DataDingsDatasource(
                                                                            data_folder, type=TYPE_LABEL,
                                                                            split=eval_split),
                                                                        dataset_name='labels')
                                                  ])
                          )
    complete_pipe = MultiSequenceModule([
        # {} -> {'features', 'labels'}
        label_pipe_cache,
        # {'features', 'labels'} -> {'features', 'labels'}
        PipelineAdaptorModule(selection_criterion=NameSelectionCriterion('features'),
                              pipe_module=SequenceModule([
                                  WhitelistModule(['VV', 'VH']),
                                  UnsupervisedSklearnAdaptorModule(thresholding,
                                                                   do_predict=False),
                                  TerminationModule()
                              ], ignore_none=True),
                              dataset_name=None, keep_source=True),
        # {'features', 'labels'} -> {'features', 'labels', 'clustering'}
        PipelineAdaptorModule(NameSelectionCriterion(name='features'),
                              SequenceModule([WhitelistModule(_config['quickshift_channels']),
                                              PerImageClusteringModule(method='quickshift',
                                                                       out_channel='clustering',
                                                                       kwargs=quickshift_args),
                                              PerImageClusteringModule(method='connectivity',
                                                                       out_channel='clustering',
                                                                       kwargs=connectivity_args),
                                              ShapelessInMemoryModule()]),
                              keep_source=True,
                              dataset_name='clustering'
                              ),
        MetricsModule(prediction_criterion=NameSelectionCriterion('clustering'),
                      label_criterion=NameSelectionCriterion('labels'),
                      source_criterion=NameSelectionCriterion('features'),
                      per_data_computation=SequenceMetricComputation([
                          ContingencyMatrixComputation('quickshift.contingency'),
                          ClusterSpatialDistributionComputation('quickshift.spatial_dist')
                      ]),
                      delete_label=False,
                      delete_prediction=False,
                      delete_source=False),
        # {'features', 'labels', 'clustering'} - > {'features', 'labels', 'clustering'}
        PerImageClusteringRefinementModule(label_criterion=NameSelectionCriterion(name='clustering'),
                                           intensity_criterion=NameSelectionCriterion(name='features'),
                                           **clustering_refinement_args),
        # {'features', 'labels', 'clustering'} - > {'features', 'labels', 'clustering'}
        PipelineAdaptorModule(NameSelectionCriterion(name='clustering'),
                              PerImageClusteringModule(method='label_normalize')),
        # {'features', 'labels', 'clustering'} - > {'features', 'labels', 'clustering'}
        PipelineAdaptorModule(NameSelectionCriterion(name='clustering'),
                              ShapelessInMemoryModule()),
        MetricsModule(prediction_criterion=NameSelectionCriterion('clustering'),
                      label_criterion=NameSelectionCriterion('labels'),
                      source_criterion=NameSelectionCriterion('features'),
                      per_data_computation=SequenceMetricComputation([
                          ContingencyMatrixComputation('refined.contingency'),
                          ClusterSpatialDistributionComputation('refined.spatial_dist')
                      ]),
                      delete_label=False,
                      delete_prediction=False,
                      delete_source=False),
        # {'features', 'labels', 'clustering'} -> {'features', 'labels', 'clustering', 'zonal_stats'}
        PerImageZonalStatsExtractorModule(label_criterion=NameSelectionCriterion(name='clustering'),
                                          intensity_criterion=NameSelectionCriterion(name='features'),
                                          dataset_result_name='zonal_stats',
                                          stats_of_interest=_config['k_means_stats'],
                                          keep_intensity=True,
                                          keep_label=True,
                                          no_stat_suffix=True),
        PipelineAdaptorModule(NameSelectionCriterion(name='zonal_stats'),
                              ShapelessInMemoryModule()),
        # {'features', 'labels', 'clustering', 'zonal_stats'} ->
        # {'features', 'labels', 'clustering', 'zonal_stats', 'zonal_clustering'}
        PipelineAdaptorModule(NameSelectionCriterion(name='zonal_stats'),
                              UnsupervisedSklearnAdaptorModule(k_means),  # do K-Means
                              keep_source=True,
                              dataset_name='zonal_clustering'),
        # {'features', 'labels', 'clustering', 'zonal_stats'} ->
        # {'features', 'labels', 'clustering', 'zonal_stats', 'zonal_clustering', 'zonal_clustering_result'}
        AssemblerModule(label_criterion=NameSelectionCriterion(name='clustering'),
                        value_criterion=NameSelectionCriterion(name='zonal_clustering'),
                        res_dataset_name='zonal_clustering_result',
                        delete_value=False,
                        delete_label=False),
        # {'features', 'labels', 'clustering', 'zonal_stats', 'zonal_clustering', 'zonal_clustering_result'} ->
        # {'labels', 'clustering', 'zonal_stats', 'zonal_clustering'}
        MetricsModule(prediction_criterion=NameSelectionCriterion('zonal_clustering_result'),
                      label_criterion=NameSelectionCriterion('labels'),
                      source_criterion=NameSelectionCriterion('features'),
                      per_data_computation=SequenceMetricComputation([
                          ContingencyMatrixComputation('k_means.contingency'),
                          ClusterSpatialDistributionComputation('kmeans.spatial_dist')
                      ]),
                      delete_label=False,
                      delete_prediction=True,
                      delete_source=False),

        # {'features', 'labels', 'clustering', 'zonal_stats', 'zonal_clustering'} ->
        # {'features', 'labels', 'clustering', 'zonal_stats', 'zonal_clustering'}
        PipelineAdaptorModule(NameSelectionCriterion(name='zonal_stats'),
                              WhitelistModule(['VV', 'VH'])),
        # {'features', 'labels', 'clustering', 'zonal_stats', 'zonal_clustering'} ->
        # {'features', 'labels', 'clustering', 'zonal_clustering', 'zonal_clustering_stats'}
        PerDataPointStatsExtractorModule(
            label_criterion=NameSelectionCriterion(name='zonal_clustering'),
            intensity_criterion=NameSelectionCriterion(name='zonal_stats'),
            stats_of_interest=['mean_intensity'],
            dataset_result_name='zonal_clustering_stats',
            keep_label=True,
            no_stat_suffix=True),
        # {'features', 'labels', 'clustering', 'zonal_clustering', 'zonal_clustering_stats'}
        # -> {'features', 'labels', 'clustering', 'zonal_clustering', 'classification'}
        PipelineAdaptorModule(NameSelectionCriterion(name='zonal_clustering_stats'),
                              SequenceModule([
                                  UnsupervisedSklearnAdaptorModule(thresholding, do_fit=False),
                                  UnsupervisedSklearnAdaptorModule(bitwise_combine, do_fit=False)
                              ]),
                              dataset_name='classification'),
        # {'features', 'labels', 'clustering', 'zonal_clustering', 'classification'} ->
        # {'features', 'labels', 'clustering', 'zonal_clustering_classification'}
        AssemblerModule(label_criterion=NameSelectionCriterion(name='zonal_clustering'),
                        value_criterion=NameSelectionCriterion(name='classification'),
                        res_dataset_name='zonal_clustering_classification'),
        # {'features', 'labels', 'clustering', 'zonal_clustering_classification'} ->
        # {'features', 'labels', 'clustering_classification'}
        AssemblerModule(label_criterion=NameSelectionCriterion(name='clustering'),
                        value_criterion=NameSelectionCriterion(name='zonal_clustering_classification'),
                        res_dataset_name='clustering_classification'),
        MetricsModule(prediction_criterion=NameSelectionCriterion('clustering_classification'),
                      label_criterion=NameSelectionCriterion('labels'),
                      per_data_computation=ContingencyMatrixComputation('final.contingency'),
                      delete_label=True,
                      delete_prediction=True)
    ])
    return complete_pipe


def pipeline(data_folder: str, cons_by_filter: Dict[str, Dict[str, Tuple[Pipeline, Pipeline]]],
             eval_split: str = SPLIT_VALIDATION) -> Callable:

    pipe_cache = SimpleCachedPipeline()
    @trial_wrapper
    def main(_config, _seed, _hook):
        print('--------------------------------------------------------------------------')
        complete_pipe = raw_pipeline(data_folder, eval_split, pipe_cache, _config, _seed, cons_by_filter)
        print('Serializing pipeline')
        with _hook.open_artifact_file('pipeline.json', 'w') as fd:
            serialize(fd, complete_pipe)
        print('Serialisation completed to file pipeline.json')
        summary = Summary([], lambda a: '')
        summary.set_hook(_hook.add_result_metric_prefix('final.contingency'))
        print('Starting Pipeline')
        t = time.time()
        res_summary = complete_pipe(summary)
        t = time.time() - t
        print(f'Pipeline completed in {t:.3f}s')

        final_contingency_matrices = [d[0] for k, d in _hook.recorded_metrics.items() if
                                      k.startswith('final.contingency')]
        accuracy_values = [accuracy_from_contingency(d) for d in final_contingency_matrices]
        res = sum(map(lambda t: t[0], accuracy_values)) / sum(map(lambda t: t[1], accuracy_values))
        print(f'Mean Overall Accuracy is {res}.')
        return res

    return main


def execute_grid_search_experiments(data_folder: str, experiment_base_folder: str,
                                    use_datadings: bool = True, seed: Optional[int] = None,
                                    timeout: Optional[int] = None,
                                    filters: Optional[List[str]] = None,
                                    redirect_output: bool =True):
    ex_name = NAME + '_2_grid_search'
    print('Constructing feature spaces.')
    cons_by_filter = default_feature_space_construction(data_folder, filters, use_datadings=use_datadings)
    print(f'Construction complete.')

    def config(trial: ConditionedTrial):
        selected_filter = trial.suggest_categorical('selected_filter', all_filters, selected_choices=selected_filters,
                                                    condition=True)
        fs = trial.suggest_categorical('feature_space', f_names[selected_filter],
                                       selected_choices=['SAR', 'SAR+R', 'SAR_O3', 'SAR_OPT', 'SAR+R_O3', 'SAR+R_OPT'],
                                       condition=True)
        available_channels = []
        if 'SAR' in fs:
            available_channels.append(['VV', 'VH'])
            if 'SAR+R' in fs:
                available_channels.append(['VV', 'VH', 'VV-VH lin. Ratio'])
        if 'NDWI' in fs:
            available_channels.append(['NDWI', 'MNDWI1'])
            if 'NDVI' in fs:
                available_channels.append(['NDWI', 'MNDWI1', 'NDVI'])
        if 'AWEI' in fs:
            available_channels.append(['AWEI', 'AWEISH'])
            if 'NDVI' in fs:
                available_channels.append(['AWEI', 'AWEISH', 'NDVI'])
            if 'NDWI' in fs:
                available_channels.append(['AWEI', 'AWEISH', 'NDWI', 'MNDWI1'])
                if 'NDVI' in fs:
                    available_channels.append(['AWEI', 'AWEISH', 'NDWI', 'MNDWI1', 'NDVI'])
        trial.suggest_categorical('quickshift_channels', available_channels,
                                  selected_choices=[['VV', 'VH']])

        trial.suggest_uniform('quickshift_ratio', 0.0, 1.0, selected_choices=[1.0])
        trial.suggest_int('quickshift_kernel_size', 7, 37, selected_choices=[7])
        trial.suggest_uniform('quickshift_max_dist', 1.0, 10.0, selected_choices=[4.0])
        trial.suggest_uniform('quickshift_sigma', 0.0, 5.0, selected_choices=[0.0])

        trial.suggest_int('post_quickshift_connectivity', 1, 2, selected_choices=[1])

        trial.suggest_categorical('k_means_stats', [['mean_intensity']])  # , ['mean_intensity', 'standard_deviation']])
        trial.suggest_int('k_means_clusters', 2, 100, selected_choices=[i for i in range(2, 16)])

        trial.suggest_int('clustering_refinement_connectivity', 1, 2, selected_choices=[1])
        trial.suggest_int('clustering_refinement_convergence_threshold', 1, 100, selected_choices=[10])
        trial.suggest_uniform('clustering_refinement_perim_area_threshold', 3.0, 48.0,
                              selected_choices=[8.0, 12.0, 16.0])
        trial.suggest_uniform('clustering_refinement_rel_std_threshold', 0.25, 4.0, selected_choices=[1.0])
        trial.suggest_int('clustering_refinement_max_iter', 0, 20, selected_choices=[10])
        trial.suggest_categorical('clustering_refinement_remove_singular',
                                  choices=[True, False],
                                  selected_choices=[True])
        greedy = trial.suggest_categorical('clustering_refinement_greedy_strategy',
                                           choices=[True, False],
                                           selected_choices=[True],
                                           condition=True)
        if not greedy:
            trial.suggest_int('clustering_refinement_max_merge_trials', 1, 1000, selected_choices=[100])
        trial.leave_condition('clustering_refinement_greedy_strategy')

        trial.suggest_categorical('threshold_alg', ['otsu', 'ki'])
        trial.suggest_categorical('bin_count', ['tile_dim', 'auto'])
        use_tiled = trial.suggest_categorical('threshold_use_tiled', [True], condition=True)  # [True, False]
        if use_tiled:
            trial.suggest_categorical('threshold_tile_dim', [32, 64, 128, 256], selected_choices=[256])
            trial.suggest_categorical('threshold_force_merge_on_failure', [True, False], selected_choices=[False])
            trial.suggest_float('threshold_percentile', 0.8, 0.99, selected_choices=[0.95])
        trial.leave_condition(['threshold_use_tiled','feature_space','selected_filter'])


    run_sacred_grid_search(experiment_base_folder, ex_name, pipeline(data_folder, cons_by_filter), config, seed, timeout=timeout,
                           redirect_output=redirect_output, direction=optuna.study.StudyDirection.MAXIMIZE)


def execute_final_experiments(data_folder: str, experiment_base_folder: str,
                              use_datadings: bool = True, seed: Optional[int] = None,
                              timeout: Optional[int] = None,
                              filters: Optional[List[str]] = None,
                              redirect_output: bool = True):
    ex_name = NAME + '_final'
    for split in [SPLIT_TEST, SPLIT_BOLIVIA]:
        cons_by_filter, f_names, all_filters, selected_filters = default_feature_space_construction(data_folder, filters,
                                                                                                use_datadings, eval_split=split)

        def config(trial: ConditionedTrial):
            selected_filter = trial.suggest_categorical('selected_filter', all_filters,
                                                        selected_choices=selected_filters,
                                                        condition=True)
            fs = trial.suggest_categorical('feature_space', f_names[selected_filter],
                                           selected_choices=['SAR'],
                                           condition=True)
            available_channels = []
            if 'SAR' in fs:
                available_channels.append(['VV', 'VH'])
                if 'SAR+R' in fs:
                    available_channels.append(['VV', 'VH', 'VV-VH lin. Ratio'])
            if 'NDWI' in fs:
                available_channels.append(['NDWI', 'MNDWI1'])
                if 'NDVI' in fs:
                    available_channels.append(['NDWI', 'MNDWI1', 'NDVI'])
            if 'AWEI' in fs:
                available_channels.append(['AWEI', 'AWEISH'])
                if 'NDVI' in fs:
                    available_channels.append(['AWEI', 'AWEISH', 'NDVI'])
                if 'NDWI' in fs:
                    available_channels.append(['AWEI', 'AWEISH', 'NDWI', 'MNDWI1'])
                    if 'NDVI' in fs:
                        available_channels.append(['AWEI', 'AWEISH', 'NDWI', 'MNDWI1', 'NDVI'])
            trial.suggest_categorical('quickshift_channels', available_channels,
                                      selected_choices=[['VV', 'VH']])

            trial.suggest_uniform('quickshift_ratio', 0.0, 1.0, selected_choices=[1.0])
            trial.suggest_int('quickshift_kernel_size', 7, 37, selected_choices=[7])
            trial.suggest_uniform('quickshift_max_dist', 1.0, 10.0, selected_choices=[4.0])
            trial.suggest_uniform('quickshift_sigma', 0.0, 5.0, selected_choices=[0.0])

            trial.suggest_int('post_quickshift_connectivity', 1, 2, selected_choices=[1])

            trial.suggest_categorical('k_means_stats',
                                      [['mean_intensity']])  # , ['mean_intensity', 'standard_deviation']])
            trial.suggest_int('k_means_clusters', 2, 100, selected_choices=[3])

            trial.suggest_int('clustering_refinement_connectivity', 1, 2, selected_choices=[1])
            trial.suggest_int('clustering_refinement_convergence_threshold', 1, 100, selected_choices=[10])
            trial.suggest_uniform('clustering_refinement_perim_area_threshold', 3.0, 48.0,
                                  selected_choices=[12.0])
            trial.suggest_uniform('clustering_refinement_rel_std_threshold', 0.25, 4.0, selected_choices=[1.0])
            trial.suggest_int('clustering_refinement_max_iter', 0, 20, selected_choices=[10])
            trial.suggest_categorical('clustering_refinement_remove_singular',
                                      choices=[True, False],
                                      selected_choices=[True])
            greedy = trial.suggest_categorical('clustering_refinement_greedy_strategy',
                                               choices=[True, False],
                                               selected_choices=[True],
                                               condition=True)
            if not greedy:
                trial.suggest_int('clustering_refinement_max_merge_trials', 1, 1000, selected_choices=[100])
            trial.leave_condition('clustering_refinement_greedy_strategy')

            trial.suggest_categorical('threshold_alg', ['otsu', 'ki'])
            trial.suggest_categorical('bin_count', ['tile_dim', 'auto'], selected_choices=['tile_dim'])
            use_tiled = trial.suggest_categorical('threshold_use_tiled', [True], condition=True)  # [True, False]
            if use_tiled:
                trial.suggest_categorical('threshold_tile_dim', [32, 64, 128, 256], selected_choices=[256])
                trial.suggest_categorical('threshold_force_merge_on_failure', [True, False], selected_choices=[False])
                trial.suggest_float('threshold_percentile', 0.8, 0.99, selected_choices=[0.95])
            trial.leave_condition(['threshold_use_tiled', 'feature_space', 'selected_filter'])

        try:
            run_sacred_grid_search(experiment_base_folder, ex_name+'_'+split, pipeline(data_folder, cons_by_filter, eval_split=split),
                                   config, seed,
                                   timeout=timeout,
                                   redirect_output=redirect_output, direction=optuna.study.StudyDirection.MAXIMIZE)
        except TrialAbort:
            print('Execution was aborted. Trying next experiment!', file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)