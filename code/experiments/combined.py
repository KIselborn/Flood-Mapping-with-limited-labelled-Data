import optuna
from metrics import *
from optuna_sacred_adaptor import ConditionedTrial, run_sacred_grid_search, trial_wrapper, accuracy_from_contingency, TrialAbort
from pre_process import *
from serialisation import *
import experiments.mean_shift as mean_shift
import experiments.segmentation_clustering as seg_cluster
import experiments.kmeans as kmeans_cluster
import experiments.simple_methods as simple_classifiers
import experiments.discriminant_analysis as discriminant_analysis
import experiments.simple_deterministic_methods as deterministic_classifiers

NAME = 'combined'

# used for visualisation - see visualize_images
def raw_pipeline(data_folder: str, eval_split: str, label_pipe_cache: SimpleCachedPipeline,
                 feature_pipe_cache: SimpleCachedPipeline, _config: Dict[str, Any], _seed: int,
                 cons_by_filter: Dict[str, Dict[str, Tuple[Pipeline, Pipeline]]],
                 cluster_cons_by_filter: Dict[str, Dict[str, Tuple[Pipeline, Pipeline]]],
                 include_save_operations: bool = True) -> MultiPipeline:
    classifier_feature_space = None
    if _config['classification_method'] in ['GB-RF', 'SGDClassifier']:
        _config['method'] = _config['classification_method']
        if _config['classification_method'] == 'GB-RF':
            feature_space = _config['feature_space_gb']
            classifier_feature_space = feature_space
            _config['z_feature_space'] = feature_space
            _config['num_leaves'] = _config[feature_space+'_num_leaves']
            _config['n_estimators'] = _config[feature_space+'_n_estimators']
            _config['subsample_for_bin'] = _config[feature_space+'_subsample_for_bin']
            _config['reg_lambda'] = _config[feature_space+'_reg_lambda']
            _config['class_weight'] = _config[feature_space+'_class_weight_gb']
        else:
            feature_space = _config['feature_space_sgd']
            classifier_feature_space = feature_space
            _config['z_feature_space'] = feature_space
            _config['loss'] = _config[feature_space+'_loss']
            _config['alpha'] = _config[feature_space+'_alpha']
            _config['class_weight'] = _config[feature_space+'_class_weight']
        classifier_pipe, classifier_factory = simple_classifiers.raw_pipeline(data_folder, eval_split, NoCachePipeline(),
                                                          NoCachePipeline(), _config, _seed, cons_by_filter,
                                                                              include_save_operations=False,
                                                                              predict_factory=True)
    elif _config['classification_method'] in ['linear', 'quadratic', 'naive']:
        _config['method'] = _config['classification_method']
        feature_space = _config[_config['classification_method']+'_z_feature_space']
        classifier_feature_space = feature_space
        _config['z_feature_space'] = feature_space
        if _config['classification_method'] == 'linear':
            _config['solver'] = _config[feature_space+'solver']
            if feature_space+'_shrinkage' in _config:
                _config['shrinkage'] = _config[feature_space+'shrinkage']
        elif _config['classification_method'] == 'quadratic':
            _config['reg_param'] = _config[feature_space+'reg_param']
        classifier_pipe, classifier_factory = discriminant_analysis.raw_pipeline(data_folder, eval_split, NoCachePipeline(),
                                                          NoCachePipeline(), _config, _seed, cons_by_filter,
                                                                              include_save_operations=False,
                                                                              predict_factory=True)
    elif _config['classification_method'] == 'threshold':
        _config['SAR_method'] = _config['classification_method']
        classifier_feature_space = 'SAR'
        classifier_pipe, classifier_factory = deterministic_classifiers.raw_pipeline(data_folder, eval_split, NoCachePipeline(),
                                                          NoCachePipeline(), _config, _seed, cons_by_filter,
                                                                              include_save_operations=False,
                                                                              predict_factory=True)
    else:
        raise RuntimeError
    _config['feature_space'] = _config[_config['cluster_alg'] +  'feature_space']
    if _config['cluster_alg'] == 'mean_shift':
        _config['feature_space'] = _config[_config['cluster_alg']+'feature_space']
        cluster_pipe = mean_shift.raw_pipeline(data_folder, eval_split, NoCachePipeline(), NoCachePipeline(), _config, _seed,
                                               cluster_cons_by_filter,False)
    elif _config['cluster_alg'] in ['Gaussian Mixture', 'Quickshift']:
        feature_space = _config[_config['cluster_alg']+'feature_space']
        _config['feature_space'] = feature_space
        cluster_pipe = seg_cluster.raw_pipeline(data_folder, eval_split, NoCachePipeline(), NoCachePipeline(), _config, _seed,
                                               cluster_cons_by_filter, False)
    elif _config['cluster_alg'] == 'k-Means':
        feature_space = _config[_config['cluster_alg']+'feature_space']
        _config['feature_space'] = feature_space
        cluster_pipe = kmeans_cluster.raw_pipeline(data_folder, eval_split, NoCachePipeline(), NoCachePipeline(), _config,
                                                   _seed,cons_by_filter, False)
    else:
        raise RuntimeError()
    seq = [MultiDistributorModule([
               MultiSequenceModule([cluster_pipe, RetainInSummaryModule(NameSelectionCriterion(name='valid_clustering'))]),
               MultiSequenceModule([classifier_pipe, RetainInSummaryModule(NameSelectionCriterion(name='valid_prediction'))]),
               PipelineAdaptorModule(selection_criterion=None,
                                      pipe_module=Sen1Floods11DataDingsDatasource(data_folder, type=TYPE_LABEL,
                                                                              split=eval_split,
                                                                              as_array=True),
                                      dataset_name='valid_labels')
           ]),

    ]
    if _config['combine_type'] == 'statistic':
        seq.extend([
           PipelineAdaptorModule(selection_criterion=None,
                                  pipe_module=cons_by_filter[_config['filter']][classifier_feature_space][1],
                                  dataset_name='valid_features'),
            PerImageZonalStatsExtractorModule(label_criterion=NameSelectionCriterion(name='valid_clustering'),
                                              intensity_criterion=NameSelectionCriterion(name='valid_features'),
                                              dataset_result_name='valid_features',
                                              stats_of_interest=['mean_intensity'],
                                              keep_intensity=False,
                                              keep_label=True,
                                              no_stat_suffix=True),
            PipelineAdaptorModule(NameSelectionCriterion(name='valid_features'),
                                  ShapelessInMemoryModule(),
                                  dataset_name='valid_features'),
            RemoveFromSummaryModule(NameSelectionCriterion(name='valid_prediction')),
            classifier_factory(NameSelectionCriterion(name='valid_features')),
            AssemblerModule(label_criterion=NameSelectionCriterion(name='valid_clustering'),
                            value_criterion=NameSelectionCriterion(name='valid_prediction'),
                            res_dataset_name='valid_prediction')
        ])
    elif _config['combine_type'] == 'majority':
        seq.extend([
            PerImageZonalStatsExtractorModule(label_criterion=NameSelectionCriterion(name='valid_clustering'),
                                              intensity_criterion=NameSelectionCriterion(name='valid_prediction'),
                                              dataset_result_name='valid_prediction',
                                              stats_of_interest=['majority'],
                                              keep_intensity=False,
                                              keep_label=True,
                                              no_stat_suffix=True),
            PipelineAdaptorModule(NameSelectionCriterion(name='valid_prediction'),
                                  ShapelessInMemoryModule(),
                                  dataset_name='valid_prediction'),
            AssemblerModule(label_criterion=NameSelectionCriterion(name='valid_clustering'),
                            value_criterion=NameSelectionCriterion(name='valid_prediction'),
                            res_dataset_name='valid_prediction')
        ])
    if include_save_operations:
        seq.append(MetricsModule(prediction_criterion=NameSelectionCriterion('valid_prediction'),
                                 label_criterion=NameSelectionCriterion('valid_labels'),
                                 per_data_computation=ContingencyMatrixComputation('valid.contingency'),
                                 delete_prediction=False))
    complete_pipe = MultiSequenceModule(seq)
    return complete_pipe

def pipeline(data_folder: str, cons_by_filter: Dict[str, Dict[str, Tuple[Pipeline, Pipeline]]], cluster_cons_by_filter: Dict[str, Dict[str, Tuple[Pipeline, Pipeline]]],
             eval_split: str = SPLIT_VALIDATION) -> Callable:
    feature_pipe_cache = SimpleCachedPipeline()
    label_pipe_cache = SimpleCachedPipeline()

    @trial_wrapper
    def main(_config, _seed, _hook):
        complete_pipe = raw_pipeline(data_folder, eval_split, label_pipe_cache, feature_pipe_cache, _config,
                                     _seed, cons_by_filter, cluster_cons_by_filter)
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
        accuracy_values = [accuracy_from_contingency(d) for d in final_contingency_matrices]
        res = sum(map(lambda t: t[0], accuracy_values)) / sum(map(lambda t: t[1], accuracy_values))
        print(f'Mean Overall Accuracy is {res}.')
        return res

    return main

# TODO check best parameters
def execute_final_experiments(data_folder: str, experiment_base_folder: str,
                              use_datadings: bool = True, seed: Optional[int] = None,
                              timeout: Optional[int] = None,
                              filters: Optional[List[str]] = None,
                              redirect_output: bool = True,
                              cluster_algs: Iterable[str] = ('k-Means', 'Quickshift', 'Gaussian Mixture', 'mean_shift')):
    ex_name = NAME + '_final_'+'_'.join(cluster_algs)
    for split in [SPLIT_VALIDATION, SPLIT_TEST, SPLIT_BOLIVIA]:
        cons_by_filter, f_names, all_filters, selected_filters = default_feature_space_construction(data_folder, filters,
                                                                                                use_datadings, eval_split=split)
        cluster_cons_by_filter, cluster_f_names, cluster_all_filters, cluster_selected_filters = \
            default_clustering_feature_space_construction(data_folder, filters, use_datadings, eval_split=split)

        def config(trial: ConditionedTrial):
            trial.suggest_categorical('combine_type', ['majority', 'statistic'])
            filter = trial.suggest_categorical('filter', all_filters, selected_choices=selected_filters, condition=True)
            # prefix with z so that it will be iterated last and we can thus maximise cache hits
            # [SAR, OPT, O3, S2, RGB, RGBN, HSV(RGB), HSV(O3), cNDWI, cAWEI, cAWEI+cNDWI, HSV(O3)+cAWEI+cNDWI, SAR_OPT, SAR_O3, SAR_S2, SAR_RGB, SAR_RGBN, SAR_HSV(RGB), SAR_HSV(O3), SAR_cNDWI, SAR_cAWEI, SAR_cAWEI+cNDWI, SAR_HSV(O3)+cAWEI+cNDWI]
            method = trial.suggest_categorical('classification_method', ['GB-RF', 'SGDClassifier', 'threshold', 'linear'], condition=True)
            if method == 'GB-RF':
                feature_space = trial.suggest_categorical('feature_space_gb', f_names[filter],
                                                          selected_choices=['SAR', 'SAR_HSV(O3)+cAWEI+cNDWI',
                                                                            'HSV(O3)+cAWEI+cNDWI'],
                                                          condition=True)  # )
                boosting_type = trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss', 'rf'],
                                                          selected_choices=['gbdt'])
                # feature spaces with more than 9 features
                if feature_space in ['OPT', 'S2', 'SAR_OPT', 'SAR_S2', 'SAR_HSV(O3)+cAWEI+cNDWI']:
                    leaf_choices = [128]#[32, 64, 128]
                elif feature_space in ['SAR', 'cNDWI', 'cAWEI']:
                    leaf_choices = [2]#[2, 4]
                # feature spaces with at most 3 features
                elif feature_space in ['SAR', 'O3', 'RGB', 'HSV(RGB)', 'HSV(O3)', 'cNDWI', 'cAWEI']:  # , 'cNDWI+NDVI']:
                    leaf_choices = [4, 8]
                # 4 features
                elif feature_space in ['SAR_cNDWI', 'SAR_cAWEI', 'RGBN', 'cAWEI+cNDWI']:
                    leaf_choices = [4, 8, 16]
                # 5 features
                elif feature_space in ['SAR_O3', 'SAR_RGB', 'SAR_HSV(RGB)', 'SAR_HSV(O3)']:  # , 'SAR_cNDWI+NDVI']:
                    leaf_choices = [8, 16, 32]
                # 6 or 7 features
                elif feature_space in ['SAR_RGBN', 'SAR_cAWEI+cNDWI', 'HSV(O3)+cAWEI+cNDWI']:
                    leaf_choices = [64]#[16, 32, 64]
                else:
                    raise ValueError(f'Unknown search space {feature_space}')
                num_leaves = trial.suggest_int(feature_space + '_num_leaves', 1, 1000, selected_choices=leaf_choices)
                max_depth = trial.suggest_int('max_depth', -1, 1000, selected_choices=[-1])
                learning_rate = trial.suggest_uniform('learning_rate', 1e-7, 1e3, selected_choices=[0.1])
                n_estimators = trial.suggest_int(feature_space+'_n_estimators', 1, 1000, selected_choices=[(50 if feature_space == 'SAR' else 200)])
                subsample_for_bin = trial.suggest_int(feature_space+'_subsample_for_bin', 1, 1_000_000_000,
                                                      selected_choices=([4 * 512 * 512]
                                                                        if feature_space == 'HSV(O3)+cAWEI+cNDWI'
                                                                        else [512 * 512]))
                class_weight = trial.suggest_categorical(feature_space+'_class_weight_gb', [None])
                min_split_gain = trial.suggest_uniform('min_split_gain', 0.0, 1000., selected_choices=[0.0])
                min_child_weight = trial.suggest_uniform('min_child_weight', 0.0, 1e6, selected_choices=[50.0])
                min_child_samples = trial.suggest_int('min_child_samples', 1, 1_000_000, selected_choices=[100])
                reg_alpha = trial.suggest_uniform('reg_alpha', 0.0, 10.0, selected_choices=[0.0])
                reg_lambda = trial.suggest_uniform(feature_space+'_reg_lambda', 0.0, 10.0,
                                                   selected_choices=([1.0]))
                trial.leave_condition('feature_space_gb')
            elif method == 'SGDClassifier':
                feature_space = trial.suggest_categorical('feature_space_sgd', f_names[filter],
                                                          selected_choices=['SAR', 'SAR_HSV(O3)',
                                                                            'HSV(O3)'],
                                                          condition=True)  # )
                loss = trial.suggest_categorical(feature_space+'_loss', ['hinge', 'log', 'modified_huber', 'squared_hinge'],
                                                 selected_choices=(['hinge'] if feature_space == 'SAR' else ['log']),
                                                 condition=True)
                if loss in ['hinge', 'modified_huber', 'squared_hinge']:
                    epsilon = trial.suggest_uniform('epsilon', 1e-7, 1e3, selected_choices=[0.1])
                trial.leave_condition(feature_space+'_loss')
                alpha = trial.suggest_uniform(feature_space+'_alpha', 1e-7, 1e3,
                                              selected_choices=([0.1] if feature_space == 'SAR' else [0.0001]))
                penalty = trial.suggest_categorical('penalty', ['l2', 'l1', 'elasticnet'],
                                                    selected_choices=['l2'],
                                                    condition=True)
                if penalty == 'elasticnet':
                    l1_ratio = trial.suggest_uniform('l1_ratio', 0, 1, selected_choices=[0.15, 0.5])
                trial.leave_condition('penalty')

                learning_rate = trial.suggest_categorical('sgd_learning_rate',
                                                          ['constant', 'optimal', 'invscaling', 'adaptive']
                                                          , selected_choices=['adaptive'], condition=True)
                if learning_rate in ['constant', 'invscaling', 'adaptive']:
                    eta0 = trial.suggest_uniform('eta0', 1e-8, 1e3, selected_choices=[1e-4])
                trial.leave_condition('sgd_learning_rate')

                class_weight = trial.suggest_categorical(feature_space+'_class_weight',
                                                         (['balanced'] if feature_space == 'SAR' else [None]))
                max_iter = trial.suggest_int('max_iter', 1, 1000, selected_choices=[20])
                trial.leave_condition('feature_space_sgd')
            elif method == 'threshold':
                trial.suggest_categorical('z_feature_space', ['SAR'])
                trial.suggest_categorical('threshold_alg', ['otsu'])
                trial.suggest_categorical('bin_count', ['tile_dim'], selected_choices=['tile_dim'])
                use_tiled = trial.suggest_categorical('threshold_use_tiled', [True, False], selected_choices=[False])
                target_eval = trial.suggest_categorical('target_eval', ['VH'])
            elif method == 'linear':
                feature_space = trial.suggest_categorical(method+'_z_feature_space', f_names[filter],
                                                          selected_choices=['SAR_cAWEI+cNDWI', 'cAWEI', 'SAR'],
                                                          condition=True)
                solver = trial.suggest_categorical(feature_space+'solver',
                                                   ['eigen'],
                                                   condition=True)
                if solver in ['lsqr', 'eigen']:
                    shrinkage = trial.suggest_uniform(feature_space+'shrinkage', 0.0, 1.0,
                                          selected_choices=[(0.1 if feature_space == 'SAR_cAWEI+cNDWI' else 1.0)])
                trial.leave_condition(feature_space+'solver')
            elif method == 'quadratic':
                feature_space = trial.suggest_categorical(method+'_z_feature_space', f_names[filter],
                                                          selected_choices=['SAR_HSV(O3)+cAWEI+cNDWI', 'cAWEI', 'SAR'],
                                                          condition=True)
                reg = 0.0
                if feature_space == 'cAWEI':
                    reg = 0.001
                elif feature_space == 'SAR':
                    reg = 1.0
                trial.suggest_uniform(feature_space+'reg_param', 0.0, 1e3, selected_choices=[reg])
            elif method == 'naive':
                feature_space = trial.suggest_categorical(method+'_z_feature_space', f_names[filter],
                                                          selected_choices=['SAR_HSV(O3)+cAWEI+cNDWI', 'cAWEI+cNDWI', 'SAR'],
                                                          condition=False)
            else:
                raise NotImplementedError(f'Method {method} is not supported yet')
            trial.leave_condition('classification_method')
            ############################################################
            # Clustering
            ############################################################
            # K-Means (Global)
            ###
            cluster_alg = trial.suggest_categorical('cluster_alg', cluster_algs, condition=True) # ['Gaussian Mixture', 'Quickshift']
            ###
            # Mean Shift
            ###
            if cluster_alg == 'mean_shift':
                tt = trial.suggest_categorical('tt', [0, 1, 2, 3])
                if tt == 0:
                    return {cluster_alg + 'feature_space': 'SAR',
                            'bandwidth': 0.5,
                            'bin_seeding': True,
                            'min_bin_freq': 50,
                            'cluster_all': False}
                elif tt == 1:
                    return {cluster_alg + 'feature_space': 'cAWEI',
                            'bandwidth': 0.5,
                            'bin_seeding': True,
                            'min_bin_freq': 50,
                            'cluster_all': False}
                elif tt == 2:
                    return {cluster_alg + 'feature_space': 'cAWEI+NDVI',
                            'bandwidth': 0.5,
                            'bin_seeding': True,
                            'min_bin_freq': 50,
                            'cluster_all': False}
                elif tt == 3:
                    return {cluster_alg + 'feature_space': 'SAR_cNDWI',
                            'bandwidth': 0.5,
                            'bin_seeding': True,
                            'min_bin_freq': 50,
                            'cluster_all': False}
            elif cluster_alg == 'Gaussian Mixture':
                tol = trial.suggest_uniform('tol', 1e-12, 1.0, selected_choices=[1e-3])
                feature_space = trial.suggest_categorical(cluster_alg + 'feature_space',
                                                          ['SAR_cAWEI', 'SAR_O3', 'cNDWI', 'O3', 'SAR'], condition=True)
                if feature_space == 'SAR_cAWEI':
                    cov_type = trial.suggest_categorical(feature_space + 'cov_type',
                                                         ['full', 'tied', 'diag', 'spherical'],
                                                         selected_choices=['diag'])
                    init_params = trial.suggest_categorical(feature_space + 'init_params', ['kmeans', 'random'],
                                                            selected_choices=['random'])
                    n_components = trial.suggest_int(feature_space + 'n_components', 2, 10_000, selected_choices=[2])
                elif feature_space == 'SAR_O3':
                    cov_type = trial.suggest_categorical(feature_space + 'cov_type',
                                                         ['full', 'tied', 'diag', 'spherical'],
                                                         selected_choices=['diag'])
                    init_params = trial.suggest_categorical(feature_space + 'init_params', ['kmeans', 'random'],
                                                            selected_choices=['random'])
                    n_components = trial.suggest_int(feature_space + 'n_components', 2, 10_000, selected_choices=[10])
                elif feature_space == 'cNDWI':
                    cov_type = trial.suggest_categorical(feature_space + 'cov_type',
                                                         ['full', 'tied', 'diag', 'spherical'],
                                                         selected_choices=['tied'])
                    init_params = trial.suggest_categorical(feature_space + 'init_params', ['kmeans', 'random'],
                                                            selected_choices=['kmeans'])
                    n_components = trial.suggest_int(feature_space + 'n_components', 2, 10_000, selected_choices=[2])
                elif feature_space == 'O3':
                    cov_type = trial.suggest_categorical(feature_space + 'cov_type',
                                                         ['full', 'tied', 'diag', 'spherical'],
                                                         selected_choices=['tied'])
                    init_params = trial.suggest_categorical(feature_space + 'init_params', ['kmeans', 'random'],
                                                            selected_choices=['kmeans'])
                    n_components = trial.suggest_int(feature_space + 'n_components', 2, 10_000, selected_choices=[10])
                elif feature_space == 'SAR':
                    cov_type = trial.suggest_categorical(feature_space + 'cov_type',
                                                         ['full', 'tied', 'diag', 'spherical'],
                                                         selected_choices=['diag'])
                    init_params = trial.suggest_categorical(feature_space + 'init_params', ['kmeans', 'random'],
                                                            selected_choices=['random'])
                    n_components = trial.suggest_int(feature_space + 'n_components', 2, 10_000, selected_choices=[2])
                else:
                    raise RuntimeError()
                return {'n_components': n_components, 'init_params': init_params, 'cov_type': cov_type}
            elif cluster_alg == 'Quickshift':
                feature_space = trial.suggest_categorical(cluster_alg + 'feature_space',
                                                          ['SAR_cAWEI+NDVI', 'cAWEI+NDVI', 'SAR'], condition=True)
                trial.suggest_categorical(feature_space + '_convert2lab', [True, False], selected_choices=[False])
                trial.suggest_int('quickshift_kernel_size', 7, 37, selected_choices=[13])
                trial.suggest_uniform('quickshift_max_dist', 1e-12, 10.0, selected_choices=[8.0])
                trial.suggest_uniform('quickshift_sigma', 0.0, 5.0, selected_choices=[0.0])
                if feature_space == 'cAWEI+NDVI':
                    quickshift_ratio = trial.suggest_uniform(feature_space + 'quickshift_ratio', 0.0, 1.0,
                                                             selected_choices=[1.0])
                else:
                    quickshift_ratio = trial.suggest_uniform(feature_space + 'quickshift_ratio', 0.0, 1.0,
                                                             selected_choices=[0.875])
                trial.leave_condition(['filter', 'cluster_alg'])

                return {'quickshift_ratio': quickshift_ratio}
            elif cluster_alg == 'k-Means':
                feature_space = trial.suggest_categorical(cluster_alg+'feature_space', ['SAR', 'O3', 'SAR_OPT'], condition=True)
                res = {}
                if feature_space == 'SAR':
                    res['k'] = 2
                elif feature_space == 'O3':
                    res['k'] = 700
                else:
                    res['k'] = 2
                trial.leave_condition([cluster_alg+'feature_space', 'filter'])
                return res
            else:
                raise ValueError('Unknown clustering algorithm ' + cluster_alg)


        try:
            run_sacred_grid_search(experiment_base_folder, ex_name+'_'+split,
                                   pipeline(data_folder, cons_by_filter, cluster_cons_by_filter, eval_split=split),
                                   config, seed, timeout=timeout,
                                   redirect_output=redirect_output, direction=optuna.study.StudyDirection.MAXIMIZE)
        except TrialAbort:
            print('Execution was aborted. Trying next experiment!', file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)