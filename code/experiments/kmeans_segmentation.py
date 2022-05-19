import functools as funct

import numpy as np
import optuna
from clustering import *
from metrics import *
from optuna_sacred_adaptor import ConditionedTrial, run_sacred_grid_search, trial_wrapper, accuracy_from_contingency, \
    TrialAbort
from pre_process import *
from serialisation import *

NAME = 'kmeans_segmentation_clustering'

def raw_pipeline(data_folder: str, eval_split: str, data_pipe_cache: SimpleCachedPipeline,
                 birch_result_pipe_cache: SimpleCachedPipeline, _config: Dict[str, Any], _seed: int,
                 cons_by_filter: Dict[str, Dict[str, Tuple[Pipeline, Pipeline]]],
                 include_save_operations: bool = True) -> MultiPipeline:
    label_pipe = PipelineAdaptorModule(selection_criterion=None,
                                       pipe_module=Sen1Floods11DataDingsDatasource(data_folder, type=TYPE_LABEL,
                                                                                   split=eval_split,
                                                                                   as_array=True),
                                       dataset_name='valid_labels'
                                       )
    feature_space = _config['feature_space']
    train_cons, valid_cons = cons_by_filter[_config['filter']][feature_space]
    feature_pipe = PipelineAdaptorModule(selection_criterion=None,
                                         pipe_module=valid_cons,
                                         dataset_name='valid_features'
                                         )
    data_pipe_cache.set_module(MultiDistributorModule([
        label_pipe,
        feature_pipe
    ]))
    cluster_alg = _config['cluster_alg']
    birch_result_pipe_cache.set_module(MultiSequenceModule([
        data_pipe_cache,
        SupervisedSklearnAdaptorModule(
            transformer=SupervisedSklearnAdaptor('clustering.FaissKMeans',
                                                 params={'k': _config['k'],
                                                         'use_gpu': (_config['use_gpu'] if 'use_gpu' in _config else True),
                                                         'random_state': _seed},
                                                 per_data_point=True),
            feature_criterion=NameSelectionCriterion('valid_features'),
            label_criterion=NameSelectionCriterion('valid_labels'),
            prediction_dataset_name='valid_prediction',
            prediction_channel_name='valid_prediction'
        ),
        PipelineAdaptorModule(selection_criterion=NameSelectionCriterion('valid_prediction'),
                              pipe_module=PerImageClusteringModule(method='label_normalize',
                                                                   kwargs={
                                                                       'allow_region_props_calc': False}),
                              dataset_name='valid_prediction',
                              keep_source=False),
        PipelineAdaptorModule(selection_criterion=NameSelectionCriterion('valid_prediction'),
                              pipe_module=ShapelessInMemoryModule(),
                              dataset_name='valid_prediction',
                              keep_source=False),
        PerImageZonalStatsExtractorModule(label_criterion=NameSelectionCriterion('valid_prediction'),
                                          intensity_criterion=None,
                                          stats_of_interest=['area', 'label'],
                                          dataset_result_name='birch_area',
                                          label_dataset_result_name='birch_prediction',
                                          keep_label=True),
        PerImageZonalStatsExtractorModule(label_criterion=NameSelectionCriterion('valid_prediction'),
                                          intensity_criterion=NameSelectionCriterion('valid_features'),
                                          stats_of_interest=['mean_intensity'],
                                          dataset_result_name='birch_mean',
                                          label_dataset_result_name=None,
                                          keep_label=True,
                                          keep_intensity=True),
        PipelineAdaptorModule(selection_criterion=NameSelectionCriterion('birch_area'),
                              pipe_module=ShapelessInMemoryModule(),
                              dataset_name='birch_area'),
        PipelineAdaptorModule(selection_criterion=NameSelectionCriterion('birch_prediction'),
                              pipe_module=ShapelessInMemoryModule(),
                              dataset_name='birch_prediction'),
        PipelineAdaptorModule(selection_criterion=NameSelectionCriterion('birch_mean'),
                              pipe_module=ShapelessInMemoryModule(),
                              dataset_name='birch_mean')
    ]))
    if 'Direct' in cluster_alg:
        computation_module = birch_result_pipe_cache
    else:
        if include_save_operations:
            metric_eval_pretext = MultiSequenceModule([
                birch_result_pipe_cache,
                MetricsModule(prediction_criterion=NameSelectionCriterion('valid_prediction'),
                              label_criterion=NameSelectionCriterion('valid_labels'),
                              source_criterion=None,
                              per_data_computation=ContingencyMatrixComputation('birch.contingency'),
                              delete_prediction=False)
            ])
        else:
            metric_eval_pretext = birch_result_pipe_cache
        data_assembler = AssemblerModule(value_criterion=NameSelectionCriterion('valid_sub_prediction'),
                                         label_criterion=NameSelectionCriterion('valid_prediction'),
                                         res_dataset_name='valid_prediction')
        if 'DB-Scan' in cluster_alg:
            adaptor = GeneralisedSklearnAdaptor(('clustering.SerializableDBScan', SerializableDBScan),
                                                params=dict(eps=_config['eps'], min_samples=_config['min_samples'],
                                                            metric=_config['dbscan_metric']),
                                                per_data_point=True, allow_no_fit=True)
            computation_module = MultiSequenceModule([
                metric_eval_pretext,
                GeneralisedSklearnAdaptorModule(transformer=adaptor,
                                                criteria_map=[
                                                    DataInfo(criterion=NameSelectionCriterion('birch_mean'),
                                                             param_name='X'),
                                                    DataInfo(criterion=NameSelectionCriterion('birch_area'),
                                                             param_name='sample_weight')],
                                                do_fit=False,
                                                prediction_dataset_name='valid_sub_prediction',
                                                prediction_channel_name='valid_sub_prediction'),
                PipelineAdaptorModule(selection_criterion=NameSelectionCriterion('valid_sub_prediction'),
                                      pipe_module=PerImageClusteringModule(method='label_normalize',
                                                                           kwargs={
                                                                               'allow_region_props_calc': False}),
                                      dataset_name='valid_sub_prediction',
                                      keep_source=False),
                data_assembler
            ])
        elif 'Hierarchical' in cluster_alg:
            adaptor = GeneralisedSklearnAdaptor(('clustering.SerializableAgglomerative', SerializableAgglomerative),
                                                params=dict(n_clusters=_config['hierarchical_clusters'],
                                                            affinity=_config[_config['linkage'] + '_affinity'],
                                                            linkage=_config['linkage']),
                                                per_data_point=True, allow_no_fit=True)
            computation_module = MultiSequenceModule([
                metric_eval_pretext,
                GeneralisedSklearnAdaptorModule(transformer=adaptor,
                                                criteria_map=[
                                                    DataInfo(criterion=NameSelectionCriterion('birch_mean'),
                                                             param_name='X')],
                                                do_fit=False,
                                                prediction_dataset_name='valid_sub_prediction',
                                                prediction_channel_name='valid_sub_prediction'),
                data_assembler
            ])
        else:
            raise ValueError('Unknown clustering algorithm ' + cluster_alg)

    seq = [
        computation_module
    ]
    if include_save_operations:
        seq.append(MetricsModule(prediction_criterion=NameSelectionCriterion('valid_prediction'),
                      label_criterion=NameSelectionCriterion('valid_labels'),
                      source_criterion=None,
                      per_data_computation=ContingencyMatrixComputation('valid.contingency'),
                      delete_prediction=True))
    complete_pipe = MultiSequenceModule(seq, ignore_none=True)
    return complete_pipe


def pipeline(data_folder: str, cons_by_filter: Dict[str, Dict[str, Tuple[Pipeline, Pipeline]]],
             eval_split: str = SPLIT_VALIDATION) -> Callable:
    data_pipe_cache = SimpleCachedPipeline()
    birch_result_pipe_cache = SimpleCachedPipeline()

    @trial_wrapper
    def main(_config, _seed, _hook):
        complete_pipe = raw_pipeline(data_folder, eval_split, data_pipe_cache, birch_result_pipe_cache, _config, _seed, cons_by_filter)
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
        accuracy_values = [accuracy_from_contingency(d, max_agree=True) for d in final_contingency_matrices]
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
        # 6 possibilities
        k = trial.suggest_int('k',2, 100_000, selected_choices=[2**i for i in range(10, 14)], condition=True)
        cluster_alg = trial.suggest_categorical('cluster_alg', ['Direct','Hierarchical', 'DB-Scan'],
                                                condition=True)
        if 'DB-Scan' in cluster_alg: # 7*5 = 35 possibilities
            eps = trial.suggest_uniform('eps', 1e-12, 1e3, selected_choices=[0.1 * (2.0**i) for i in range(-3, 5)])
            min_samples = trial.suggest_int('min_samples', 2, 1_000_000, selected_choices=[50]+list(range(500, 2001, 500)))
            metric = trial.suggest_categorical('dbscan_metric', ['euclidean', 'manhattan', 'cosine'],
                                               selected_choices=['euclidean'])
        elif 'Hierarchical' in cluster_alg: # 10*(3*1+1) = 40 possibilities
            hierarchical_clusters = trial.suggest_int('hierarchical_clusters', 2, 10_000, selected_choices=list(range(2,11)))
            linkage = trial.suggest_categorical('linkage', ['ward', 'complete', 'average', 'single'], condition=True)
            if linkage == 'ward':
                affinity = trial.suggest_categorical(linkage+'_affinity', ['euclidean'])
            else:
                affinity = trial.suggest_categorical(linkage+'_affinity', ['euclidean', 'manhattan', 'cosine'],
                                                     selected_choices=['euclidean'])
            trial.leave_condition('linkage')
        elif 'Direct' in cluster_alg:
            pass
        else:
            raise ValueError('Unknown clustering algorithm ' + cluster_alg)

        trial.leave_condition(['k','cluster_alg','feature_space', 'filter'])


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
            trial.suggest_categorical('use_gpu', [True, False], selected_choices=[False])
            cluster_alg = trial.suggest_categorical('cluster_alg', ['Direct', 'Hierarchical'],
                                                    condition=True)
            if cluster_alg == 'Direct':
                k = trial.suggest_int('k', 2, 100_000, selected_choices=[1024])
                feature_space = trial.suggest_categorical('feature_space', ['O3', 'SAR'])
            else:
                tt = trial.suggest_categorical('tt', [0, 1, 2])
                if tt == 0:
                    return {'feature_space': 'SAR',
                            'k': 2048,
                            'hierarchical_clusters': 2,
                            'linkage': 'average',
                            'average_affinity': 'euclidean'}
                elif tt == 1:
                    return {'feature_space': 'cAWEI',
                            'k': 1024,
                            'hierarchical_clusters': 2,
                            'linkage': 'average',
                            'average_affinity': 'euclidean'}
                elif tt == 2:
                    return {'feature_space': 'cAWEI',
                            'k': 1024,
                            'hierarchical_clusters': 10,
                            'linkage': 'average',
                            'average_affinity': 'euclidean'}

            trial.leave_condition(['k', 'cluster_alg', 'feature_space', 'filter'])

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