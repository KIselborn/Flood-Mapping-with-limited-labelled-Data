import functools as funct

import numpy as np
import optuna
from clustering import *
from metrics import *
from optuna_sacred_adaptor import ConditionedTrial, run_sacred_grid_search, trial_wrapper, TrialAbort
from pre_process import *
from serialisation import *

NAME = 'faiss_kmeans_fixed'
def raw_pipeline(data_folder: str, eval_split: str, label_pipe_cache: SimpleCachedPipeline,
                 feature_pipe_cache: SimpleCachedPipeline, _config: Dict[str, Any], _seed: int,
                 cons_by_filter: Dict[str, Dict[str, Tuple[Pipeline, Pipeline]]],
                 include_save_operations: bool = True) -> MultiPipeline:
    label_pipe = MultiSequenceModule([
        MultiDistributorModule([
            PipelineAdaptorModule(selection_criterion=None,
                                  pipe_module=Sen1Floods11DataDingsDatasource(data_folder, type=TYPE_LABEL,
                                                                              split=SPLIT_TRAIN, as_array=True),
                                  dataset_name='train_labels'
                                  ),
            PipelineAdaptorModule(selection_criterion=None,
                                  pipe_module=Sen1Floods11DataDingsDatasource(data_folder, type=TYPE_LABEL,
                                                                              split=eval_split,
                                                                              as_array=True),
                                  dataset_name='valid_labels'
                                  )
        ]),
        MaskModule(source_criterion=NameSelectionCriterion('train_labels'),
                   res_name='train_mask',
                   mask_label=-1)
    ])
    label_pipe_cache.set_module(label_pipe)
    feature_space = _config['feature_space']
    train_cons, valid_cons = cons_by_filter[_config['filter']][feature_space]
    feature_pipe = MultiDistributorModule([
        PipelineAdaptorModule(selection_criterion=None,
                              pipe_module=train_cons,
                              dataset_name='train_features'
                              ),
        PipelineAdaptorModule(selection_criterion=None,
                              pipe_module=valid_cons,
                              dataset_name='valid_features'
                              )
    ])
    feature_pipe_cache.set_module(feature_pipe)
    data_construction = MultiDistributorModule([
        feature_pipe_cache,
        label_pipe_cache
    ])
    adaptor = SupervisedSklearnAdaptor('clustering.FaissKMeans',
                                       params={'k': _config['k'],
                                               'use_gpu': True,
                                               'random_state': _seed},
                                       per_data_point=False,
                                       save_file=None,
                                       clear_on_predict=False)
    seq = [
        data_construction,
        SupervisedSklearnAdaptorModule(adaptor,
                                       feature_criterion=NameSelectionCriterion('train_features'),
                                       # not used, but I've got it here in order not to have to rewrite any code
                                       label_criterion=NameSelectionCriterion('train_labels'),
                                       mask_criterion=NameSelectionCriterion('train_mask'),
                                       do_predict=False),
        SupervisedSklearnAdaptorModule(adaptor,
                                       feature_criterion=NameSelectionCriterion('valid_features'),
                                       prediction_dataset_name='valid_clustering',
                                       prediction_channel_name='valid_clustering',
                                       do_fit=False)
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
            ul = ul[ul != -1]
            d[KEY_CONTINGENCY_MATRIX] = cm
            d[KEY_CONTINGENCY_UNIQUE_LABELS] = ul
            res = MaximumAgreementComputation([AccuracyComputation('accuracy')]).forward(step=None, final=True, **d)
            return (res['accuracy'], 1)

        final_contingency_matrices = [d[0] for k, d in _hook.recorded_metrics.items() if k.startswith('valid')]
        print(f'Found {len(final_contingency_matrices)} matrices.')
        accuracy_values = [accuracy_from_contingency(d) for d in final_contingency_matrices]
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
        train_cons = construct_feature_spaces(data_folder, split='train', in_memory=True,
                                              filter_method=filter_method, use_datadings=use_datadings,
                                              add_optical=True, as_array=True)
        valid_cons = construct_feature_spaces(data_folder, split='valid', in_memory=True,
                                              filter_method=filter_method, use_datadings=use_datadings,
                                              add_optical=True, as_array=True)
        cur_f_names = [t[0] for t in
                       train_cons]  # notice that we know that the filter does not influence the feature spaces
        print(f'Created {len(cur_f_names)} feature spaces [{", ".join(cur_f_names)}] for filter {filter_method}')
        cons_by_filter[filter_method] = {k1: (v1, v2) for (k1, v1), (k2, v2) in zip(train_cons, valid_cons)}
        f_names[filter_method] = cur_f_names
    print(f'Construction complete.')

    def config(trial: ConditionedTrial):
        filter = trial.suggest_categorical('filter', all_filters, selected_choices=selected_filters, condition=True)
        feature_space = trial.suggest_categorical('feature_space', f_names[filter], condition=True)
        # if per_data:
        #     k = trial.suggest_int('per_data_k', 1, 1_000,
        #                       selected_choices=list(range(2,11))+list(range(15,51,5))+list(range(100, 600, 100)))
        # else:
        k = trial.suggest_int('k', 1, 10_000, selected_choices=[2]+list(range(5,51,5))+list(range(100, 1000, 100))+[1024,2048])
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
        cons_by_filter, f_names, all_filters, selected_filters = default_feature_space_construction(data_folder,
                                                                                                    filters,
                                                                                                    use_datadings,
                                                                                                    eval_split=split)
        print(f'Construction complete.')

        def config(trial: ConditionedTrial):
            filter = trial.suggest_categorical('filter', all_filters, selected_choices=selected_filters, condition=True)
            feature_space = trial.suggest_categorical('feature_space', ['SAR', 'O3', 'SAR_OPT'], condition=True)
            res = {}
            if feature_space == 'SAR':
                res['k'] = 2
            elif feature_space == 'O3':
                res['k'] = 700
            else:
                res['k'] = 2
            # if per_data:
            #     k = trial.suggest_int('per_data_k', 1, 1_000,
            #                       selected_choices=list(range(2,11))+list(range(15,51,5))+list(range(100, 600, 100)))
            # else:
            trial.leave_condition(['feature_space', 'filter'])
            return res

            # if per_data:
            #     k = trial.suggest_int('per_data_k', 1, 1_000,
            #                       selected_choices=list(range(2,11))+list(range(15,51,5))+list(range(100, 600, 100)))
            # else:

        try:
            run_sacred_grid_search(experiment_base_folder, ex_name + '_' + split,
                                   pipeline(data_folder, cons_by_filter, eval_split=split),
                                   config, seed, timeout=timeout,
                                   redirect_output=redirect_output, direction=optuna.study.StudyDirection.MAXIMIZE)
        except TrialAbort:
            print('Execution was aborted. Trying next experiment!', file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)