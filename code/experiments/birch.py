import functools as funct

import numpy as np
import optuna
from clustering import *
from metrics import *
from optuna_sacred_adaptor import ConditionedTrial, run_sacred_grid_search, trial_wrapper
from pre_process import *
from serialisation import *

NAME = 'birch_segmentation_clustering'


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
        threshold = trial.suggest_float('threshold', 1e-6, 1e3, selected_choices=[0.01, 0.05, 0.1, 0.15, 0.2], condition=True, enter=False)
        branching_factor = trial.suggest_int('branching_factor', 2, 1_000_000, selected_choices=[50, 200], condition=True, enter=False)
        trial.enter_condition(['threshold', 'branching_factor'])
        cluster_alg = trial.suggest_categorical('cluster_alg', ['Direct','Hierarchical', 'DB-Scan'],
                                                condition=True)
        if 'DB-Scan' in cluster_alg:  # 7*5 = 35 possibilities
            eps = trial.suggest_uniform('eps', 1e-12, 1e3, selected_choices=[0.1 * (2.0 ** i) for i in range(-4, 3)])
            min_samples = trial.suggest_int('min_samples', 2, 1_000_000, selected_choices=list(range(50, 1051, 250)))
            metric = trial.suggest_categorical('dbscan_metric', ['euclidean', 'manhattan', 'cosine'],
                                               selected_choices=['euclidean'])
        elif 'Hierarchical' in cluster_alg:  # 10*(3*1+1) = 40 possibilities
            hierarchical_clusters = trial.suggest_int('hierarchical_clusters', 2, 10_000,
                                                      selected_choices=list(range(2, 11)))
            linkage = trial.suggest_categorical('linkage', ['ward', 'complete', 'average', 'single'], condition=True)
            if linkage == 'ward':
                affinity = trial.suggest_categorical(linkage + '_affinity', ['euclidean'])
            else:
                affinity = trial.suggest_categorical(linkage + '_affinity', ['euclidean', 'manhattan', 'cosine'],
                                                     selected_choices=['euclidean'])
            trial.leave_condition('linkage')
        elif 'Direct' in cluster_alg:
            pass
        else:
            raise ValueError('Unknown clustering algorithm ' + cluster_alg)

        # if per_data:
        #     k = trial.suggest_int('per_data_k', 1, 1_000,
        #                       selected_choices=list(range(2,11))+list(range(15,51,5))+list(range(100, 600, 100)))
        # else:
        trial.leave_condition(['cluster_alg', 'branching_factor','threshold', 'feature_space', 'filter'])

    data_pipe_cache = SimpleCachedPipeline()
    birch_result_pipe_cache = SimpleCachedPipeline()

    @trial_wrapper
    def main(_config, _seed, _hook):
        label_pipe = PipelineAdaptorModule(selection_criterion=None,
                                  pipe_module=Sen1Floods11DataDingsDatasource(data_folder, type=TYPE_LABEL,
                                                                              split=SPLIT_VALIDATION,
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
        threshold = _config['threshold']
        branching_factor = _config['branching_factor']
        birch_result_pipe_cache.set_module(MultiSequenceModule([
            data_pipe_cache,
            SupervisedSklearnAdaptorModule(
                transformer=SupervisedSklearnAdaptor('clustering.SerializableDirectBirch',
                                                     params={
                                                         'threshold': threshold,
                                                         'branching_factor': branching_factor,
                                                         'n_clusters': None
                                                     },
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
            metric_eval_pretext = MultiSequenceModule([
                birch_result_pipe_cache,
                MetricsModule(prediction_criterion=NameSelectionCriterion('valid_prediction'),
                              label_criterion=NameSelectionCriterion('valid_labels'),
                              source_criterion=None,
                              per_data_computation=ContingencyMatrixComputation('birch.contingency'),
                              delete_prediction=False)
            ])
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

        complete_pipe = MultiSequenceModule([
            computation_module,
            MetricsModule(prediction_criterion=NameSelectionCriterion('valid_prediction'),
                          label_criterion=NameSelectionCriterion('valid_labels'),
                          source_criterion=None,
                          per_data_computation=ContingencyMatrixComputation('valid.contingency'),
                          delete_prediction=True)
        ], ignore_none=True)
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

        def accuracy_from_contingency(d: dict) -> Tuple[float, int]:
            cm = d[KEY_CONTINGENCY_MATRIX]
            ul = d[KEY_CONTINGENCY_UNIQUE_LABELS]
            up = d[KEY_CONTINGENCY_UNIQUE_PREDICTIONS]
            if np.all(ul == -1):
                return (0.0, 0)
            if not np.any(ul == 0):
                ul = np.concatenate((ul, [0]))
                cm = np.concatenate((cm, np.zeros((1, cm.shape[1]), dtype=np.int32)))
            if not np.any(ul == 1):
                ul = np.concatenate((ul, [1]))
                cm = np.concatenate((cm, np.zeros((1, cm.shape[1]), dtype=np.int32)))
            cm = cm[ul != -1]
            ul = ul[ul!= -1]
            d[KEY_CONTINGENCY_MATRIX] = cm
            d[KEY_CONTINGENCY_UNIQUE_LABELS] = ul
            res = MaximumAgreementComputation([AccuracyComputation('accuracy')]).forward(step=None, final=True, **d)
            return (res['accuracy'], 1)

        final_contingency_matrices = [d[0] for k, d in _hook.recorded_metrics.items() if k.startswith('valid')]
        print(f'Found {len(final_contingency_matrices)} matrices.')
        accuracy_values = [accuracy_from_contingency(d) for d in final_contingency_matrices]
        accuracy_values = [v for v in accuracy_values if v[1] > 0]
        print(f'Found {len(accuracy_values)} non-zero accuracy values.')
        res = sum(map(lambda t: t[0], accuracy_values)) / sum(map(lambda t: t[1], accuracy_values)) if len(accuracy_values) >= 1 else 0.0
        print(f'Mean Overall Accuracy is {res}.')
        return res

    run_sacred_grid_search(experiment_base_folder, ex_name, main, config, seed, timeout=timeout,
                           redirect_output=redirect_output, direction=optuna.study.StudyDirection.MAXIMIZE)