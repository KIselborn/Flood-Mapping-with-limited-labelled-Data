import functools as funct

import optuna
from clustering import *
from metrics import *
from optuna_sacred_adaptor import ConditionedTrial, run_sacred_grid_search, trial_wrapper
from pre_process import *
from serialisation import *

NAME = 'intelx_dbscan_segmentation'


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
        valid_cons = construct_clustering_feature_spaces(data_folder, split='valid', in_memory=True,
                                              filter_method=filter_method, use_datadings=use_datadings,
                                              as_array=True, verbose=False)
        cur_f_names = [t[0] for t in
                       valid_cons]  # notice that we know that the filter does not influence the feature spaces
        print(f'Created {len(cur_f_names)} feature spaces [{", ".join(cur_f_names)}] for filter {filter_method}')
        cons_by_filter[filter_method] = {k1: v1 for k1, v1 in valid_cons}
        f_names[filter_method] = cur_f_names
    print(f'Construction complete.')

    def config(trial: ConditionedTrial):
        filter = trial.suggest_categorical('filter', all_filters, selected_choices=selected_filters, condition=True)
        feature_space = trial.suggest_categorical('feature_space', f_names[filter], condition=True)
        min_samples = trial.suggest_int('min_samples', 2, 1_000_000, selected_choices=[50, 500])
        eps = trial.suggest_uniform('eps', 1e-12, 1e3, selected_choices=[0.5, 0.2, 0.1])
        trial.leave_condition(['feature_space', 'filter'])

    feature_pipe_cache = SimpleCachedPipeline()
    label_pipe_cache = SimpleCachedPipeline()

    @trial_wrapper
    def main(_config, _seed, _hook):
        # label_pipe = PipelineAdaptorModule(selection_criterion=None,
        #                           pipe_module=Sen1Floods11DataDingsDatasource(data_folder, type=TYPE_LABEL,
        #                                                                       split=SPLIT_VALIDATION,
        #                                                                       as_array=True),
        #                           dataset_name='valid_labels'
        #                           )
        # label_pipe_cache.set_module(label_pipe)
        # feature_space = _config['feature_space']
        # train_cons, valid_cons = cons_by_filter[_config['filter']][feature_space]
        # feature_pipe = PipelineAdaptorModule(selection_criterion=None,
        #                           pipe_module=valid_cons,
        #                           dataset_name='valid_features'
        #                           )
        # feature_pipe_cache.set_module(feature_pipe)
        # data_construction = MultiDistributorModule([
        #     feature_pipe_cache,
        #     label_pipe_cache
        # ])
        module = UnsupervisedSklearnAdaptorModule(UnsupervisedSklearnAdaptor(('clustering.DirectPredictIXDBSCAN', DirectPredictIXDBSCAN),
                                                     params={
                                                        'min_samples': _config['min_samples'],
                                                        'eps': _config['eps'],
                                                         'n_jobs': 4
                                                     },
                                                     per_data_point=True,
                                                     allow_no_fit=True),
                                                    do_fit=False)
        # module = SupervisedSklearnAdaptorModule(
        #         transformer=SupervisedSklearnAdaptor(('clustering.DirectPredictIXDBSCAN', DirectPredictIXDBSCAN),
        #                                              params={
        #                                                 'min_samples': _config['min_samples'],
        #                                                 'eps': _config['eps'],
        #                                                  'n_jobs': -1
        #                                              },
        #                                              per_data_point=True),
        #                                              feature_criterion=NameSelectionCriterion('valid_features'),
        #                                              label_criterion=None,
        #                                              prediction_dataset_name='valid_prediction',
        #                                              prediction_channel_name='valid_prediction'
        #     )

        complete_pipe = MultiSequenceModule([
            PipelineAdaptorModule(selection_criterion=None,
                                  pipe_module=cons_by_filter[_config['filter']][_config['feature_space']],
                                  dataset_name='valid_features'
                                  ),
            PipelineAdaptorModule(selection_criterion=NameSelectionCriterion('valid_features'),
                                  pipe_module=module,
                                  dataset_name='valid_prediction'
                                  ),
            PipelineAdaptorModule(selection_criterion=NameSelectionCriterion('valid_prediction'),
                                  pipe_module=PerImageClusteringModule(method='label_normalize',
                                                                       kwargs={'allow_region_props_calc': False}),
                                  dataset_name='valid_prediction',
                                  keep_source=False),
            PipelineAdaptorModule(selection_criterion=None,
                                  pipe_module=Sen1Floods11DataDingsDatasource(data_folder, type=TYPE_LABEL,
                                                                              split=SPLIT_VALIDATION,
                                                                              as_array=True),
                                  dataset_name='valid_labels'
                                  ),
            MetricsModule(prediction_criterion=NameSelectionCriterion('valid_prediction'),
                          label_criterion=NameSelectionCriterion('valid_labels'),
                          source_criterion=NameSelectionCriterion('valid_features'),
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