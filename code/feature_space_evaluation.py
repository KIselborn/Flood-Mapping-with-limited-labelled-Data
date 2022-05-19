import matplotlib.colors as cl
import pandas as pd
import sklearn.metrics as met
import sklearn.svm as svm
import warnings
from sklearn.exceptions import ConvergenceWarning

from sci_analysis import analyze

from pre_process import *
from multipipeline import *
from plot_utils import *

_SAVE = True
KEY_LABELS = 'labels'
LABEL_NAMES = {
    -1 : 'Unclassified Data',
    0: 'Dry Land',
    1: 'Water'
}


def set_nth_label_visible(labels: Iterable, n: int, force_last: bool = True):
    labels = list(labels)
    for i, label in enumerate(labels):
        if i%n == 0 or (force_last and i == len(labels) - 1):
            label.set_visible(True)
        else:
            label.set_visible(False)

def set_tick_labels(ax: plt.Axes, labels: List[str], is_x: bool, n: int, force_last: bool = True):
    if is_x:
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        set_nth_label_visible(ax.get_xticklabels(), n, force_last)
    else:
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels)
        set_nth_label_visible(ax.get_yticklabels(), n, force_last)

MAX_ROWS = 1
def create_figure_with_subplots_capped(title: str, num_rows: int, num_cols: int) -> Tuple[plt.Figure, np.ndarray, int]:
    n_rows = min(num_rows, MAX_ROWS)
    return create_figure_with_subplots(title, n_rows, num_cols)

def save_and_show_figure(fig: plt.Figure, f_name: str, show: bool, layout_dividend: int):
    save_and_show(fig, f_name, show=show, layout_dividend=layout_dividend, save=_SAVE)

def create_label_maps(df: pd.DataFrame) -> pd.DataFrame:
    label_series = df[KEY_LABELS]
    label_maps = pd.DataFrame({ul: label_series == ul for ul in label_series.unique()})
    assert len(label_maps.columns.values) == 3
    return label_maps

EPS = np.sqrt(np.finfo(np.float32).eps)
def plot_heatmap(fig: plt.Figure, ax: plt.Axes, heat_map: np.ndarray,
                 x_label, y_label, extend, x_ticks, y_ticks, cmap: Optional[str]):
    #heat_map /= heat_map.max()
    if cmap is None:
        heat_map = np.log(heat_map+1)
        max = heat_map.max()
        heat_map = np.clip(heat_map, 0.0, max)/max
        im = ax.imshow(heat_map, origin='lower', extent=extend, aspect='auto')
    else:
        im = ax.imshow(heat_map, origin='lower', cmap=cmap, norm=cl.LogNorm(), extent=extend, aspect='auto')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    #set_tick_labels(ax, x_ticks, True, 10)
    #set_tick_labels(ax, y_ticks, False, 10)
    if cmap is not None:
        fig.colorbar(im, ax=ax)

def plot_heatmap_svm(axes: np.ndarray, heat_map: np.ndarray, x_points, y_points, columns: pd.DataFrame, labels: pd.Series) \
        -> Tuple[float, float, float]:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        features = np.array([[[x, y] for x in x_points] for y in y_points]).reshape((-1, 2))
        dry_weights = heat_map[:, :, 1].reshape(-1).astype(np.int64)
        flood_weights = heat_map[:, :, 2].reshape(-1).astype(np.int64)

        combined_weights = flood_weights - dry_weights
        combined_labels = np.clip(combined_weights, 0, 1)
        combined_weights[combined_weights < 0] = np.abs(combined_weights[combined_weights < 0])
        mask = np.logical_or(dry_weights > 0, flood_weights > 0)
        combined_weights = combined_weights[mask]
        combined_labels = combined_labels[mask]
        if np.unique(combined_labels).shape[0] > 1:
            features = features[mask]
            def objective(C: float) -> float:
                new_C = 10.0 ** C if C >= 0 else (1/ (10.0 ** (-C)))
                svm_model = svm.LinearSVC(loss='squared_hinge', penalty='l1', dual=False, C=new_C, random_state=42, max_iter=1_000, tol=1e-3)
                svm_model.fit(features, combined_labels, sample_weight=combined_weights)
                prediction = svm_model.predict(features)
                return met.accuracy_score(combined_labels, prediction, sample_weight=combined_weights)
            print('Calculating SVM')
            C_space = np.linspace(-4, 4, 32)
            accuracies = [objective(C) for C in C_space]
            best_C_idx = np.argmax(accuracies)
            C = C_space[best_C_idx]
            C = 10.0 ** C if C >= 0 else (1/ (10.0 ** (-C)))
            print(f'Estimated optimal C as {C}.')
            svm_model = svm.LinearSVC(loss='squared_hinge', penalty='l1', dual=False, C=C, random_state=42, max_iter=100_000, tol=1e-4)
            svm_model.fit(features, combined_labels, sample_weight=combined_weights)
            mask = labels != -1
            pred = svm_model.predict(columns[mask])

            acc = met.accuracy_score(labels[mask], pred)
            f1 = met.f1_score(labels[mask], pred)
            print('SVM calculation completed.')
            #xlim = ax.get_xlim()
            #ylim = ax.get_ylim()
            # copied from https://scikit-learn.org/stable/auto_examples/svm/plot_separating_hyperplane.html
            xx = x_points#np.linspace(xlim[0], xlim[1], 30)
            yy = y_points#np.linspace(ylim[0], ylim[1], 30)
            YY, XX = np.meshgrid(yy, xx)
            xy = np.vstack([XX.ravel(), YY.ravel()]).T
            Z = svm_model.decision_function(xy).reshape(XX.shape)

            # plot decision boundary and margins
            for ax in axes:
                ax.contour(
                    XX, YY, Z, colors="red", levels=[-1, 0, 1], alpha=0.6, linestyles=["--", "-", "--"]
                )
            angle = np.arccos(np.clip(np.dot(svm_model.coef_/ np.linalg.norm(svm_model.coef_), np.array([1, 0])),
                                      -1.0, 1.0))
            angle = 180.0*np.abs(angle)/np.pi
            if 135.0 >= angle >= 45.0:
                angle = np.abs(90.0 - angle)
            elif angle > 135.0:
                angle = 180.0 - angle
            return float(acc), float(f1), float(angle)
        else:
            print('Combination is too indecisive to calculate the svm decision boundary!!!')
            return -0.0, -0.0, -0.0


    # see also https://stackoverflow.com/questions/23876588/matplotlib-colorbar-in-each-subplot
    # divider = make_axes_locatable(ax1)
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    # fig.colorbar(im1, cax=cax, orientation='vertical')
def plot_heatmap_columns(fig: plt.Figure, ax_row: np.ndarray, label_maps: pd.DataFrame, feature_df: pd.DataFrame, n_bins: int,
                         column_x_key: str, column_y_key: str):
    print(f'Plotting heatmaps for ("{column_x_key}", "{column_y_key}").')
    column_x: pd.Series = feature_df[column_x_key]
    column_y: pd.Series = feature_df[column_y_key]
    x_edges = np.linspace(column_x.min(), column_x.max(), n_bins + 1)
    y_edges = np.linspace(column_y.min(), column_y.max(), n_bins + 1)
    extend = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
    x_points = np.convolve(x_edges, np.array([0.5, 0.5]), mode='valid')
    y_points = np.convolve(y_edges, np.array([0.5, 0.5]), mode='valid')
    x_ticks = [f'{v:.3f}' for v in x_points]
    y_ticks = [f'{v:.3f}' for v in y_points]
    heat_map = np.empty((n_bins, n_bins, 3), dtype=np.float64)
    for i, ul, cmap in zip(range(3), sorted(label_maps.columns.values), ['Reds', 'Greens', 'Blues']):
        print(f'Calculating for label {ul}')
        column_x_used: pd.Series = column_x[label_maps[ul]]
        column_y_used: pd.Series = column_y[label_maps[ul]]
        hist, _, _ = np.histogram2d(column_y_used, column_x_used, bins=(y_edges, x_edges))
        # log_hist = np.log(hist+EPS)
        heat_map[:, :, i] = hist
        # hist = exposure.equalize_hist(hist)
        ax_row[i].set_title(f'Heatmap for {LABEL_NAMES[ul]}')
        plot_heatmap(fig, ax_row[i], hist, column_x_key, column_y_key, extend, x_ticks, y_ticks, cmap)
    ax_row[-2].set_title(f'Combined Heatmap')
    plot_heatmap(fig, ax_row[-2], heat_map, column_x_key, column_y_key, extend, x_ticks, y_ticks, None)

    acc, f1, angle = plot_heatmap_svm(ax_row, heat_map, x_points, y_points, feature_df[[column_x_key, column_y_key]],
                                      feature_df[KEY_LABELS])
    ax_row[-1].set_title(f'Flood-No-Flood Heatmap Acc.:{acc*100:.2f}% F1:{f1*100:.2f} Angle:{angle:.1f}')
    heat_map[:, :, 0] = 0.0
    plot_heatmap(fig, ax_row[-1], heat_map, column_x_key, column_y_key, extend, x_ticks, y_ticks, None)

def plot_2d_combinations(save_folder: str, df: pd.DataFrame, show: bool, n_bins: int):
    ensure_is_dir(save_folder)
    label_maps = create_label_maps(df)

    combinations = list(itert.combinations(filter(lambda v: v!= KEY_LABELS, df.columns.values), 2))
    fig, axs, n_rows = create_figure_with_subplots_capped(f'Heatmaps for 2D combinations',
                                                          len(combinations),
                                                          5)
    n_parts = 1
    save_name = path.join(save_folder, 'heatmaps')
    for i, (column_x_key, column_y_key) in enumerate(combinations):
        if i > 0 and i % MAX_ROWS == 0:
            prev_column_x_key, prev_column_y_key = combinations[i-1]
            save = save_name+f'_{n_parts}' if MAX_ROWS > 1 else save_name+f'_{prev_column_x_key}_{prev_column_y_key}'
            save_and_show_figure(fig, save, show, n_rows)
            fig, axs, n_rows = create_figure_with_subplots_capped(f'Heatmaps for 2D combinations',
                                                                  len(combinations) - i,
                                                                  5)
            n_parts += 1
        plot_heatmap_columns(fig, axs[i % MAX_ROWS], label_maps, df, n_bins, column_x_key, column_y_key)

    save = save_name + f'_{n_parts}' if MAX_ROWS > 1 else save_name + f'_{combinations[-1][0]}_{combinations[-1][1]}'
    save_and_show_figure(fig, save, show, n_rows)

def sci_analyze_groups(feature: str, save_folder: str, grouped_dict: Dict[int, pd.DataFrame]):
    print(f'Analysis of feature {feature}:')
    sys.stdout.flush()
    prev_stdout = sys.stdout
    save_file = path.join(save_folder, feature + '_location')
    try:
        with open(save_file + '.txt', 'w') as fd:
            sys.stdout = fd
            analyze({k: t[feature] for k, t in grouped_dict.items()},
                    circles=False,
                    nqp=True,
                    title=f'Location test of {feature}',
                    yname=feature,
                    xname='Flood state',
                    save_to=save_file + '.pdf')
            fd.flush()
            fd.close()
    except Exception as e:
        sys.stdout = prev_stdout
        raise RuntimeError(f'Something went wrong when analysing feature {feature}. '
                           f'Should have been saved to {save_file}') from e

def box_plot_groups(feature: str, save_folder: str, grouped_dict: Dict[int, pd.DataFrame], show: bool):
    fig, axs, n_rows = create_figure_with_subplots_capped(f'Boxplots for {feature}',
                                                          1, 2)
    axs = axs[0]
    data_boxes: plt.Axes = axs[0]
    data_boxes.boxplot([t[feature] for t in grouped_dict.values()],
                       labels=[LABEL_NAMES[l] for l in grouped_dict.keys()],
                       whis=(0., 100.))
    data_boxes.set_xlabel('Class')
    data_boxes.set_ylabel(feature)
    flood_no_flood: plt.Axes = axs[1]
    flood_no_flood.boxplot([t[feature] for l, t in grouped_dict.items() if l>=0],
                       labels=[LABEL_NAMES[l] for l in grouped_dict.keys() if l>=0],
                       whis=(0., 100.))
    flood_no_flood.set_xlabel('Class')
    flood_no_flood.set_ylabel(feature)
    save_and_show_figure(fig, path.join(save_folder, feature + '_boxplot'), show, n_rows)


def perform_1d_group_analysis(save_folder: str, df: pd.DataFrame, show: bool, include_analysis: bool):
    print('Performing group analysis')
    grouped_dict = dict(list(df.groupby(KEY_LABELS)))
    for feature in df.columns.values:
        if feature == KEY_LABELS:
            pass
        if include_analysis:
            sci_analyze_groups(feature, save_folder, grouped_dict)
        else:
            box_plot_groups(feature, save_folder, grouped_dict, show)


def perform_1d_bivariate_analysis(save_folder: str, df: pd.DataFrame, include_analysis: bool):
    if not include_analysis:
        return
    print('Performing bivariate analysis')
    label_maps = create_label_maps(df)
    labels = df[KEY_LABELS]
    df = df.drop(KEY_LABELS)
    for feature in df.columns.values:
        f_series = df[feature]
        print(f'Analysis of feature {feature}:')
        analyze(f_series[~label_maps[-1]], labels[~label_maps[-1]],
                contours=True,
                fit=False,
                title=f'Flood-No-Flood plot of {feature}',
                yname='Flood state',
                xname=feature,
                save_to=path.join(save_folder, feature+'_bivariate_filtered.pdf'))
        analyze(f_series, labels,
                contours=True,
                fit=False,
                title=f'Bivariate plot of {feature}',
                yname='Flood state',
                xname=feature,
                save_to=path.join(save_folder, feature+'_bivariate.pdf'))

def plot_1d_histogram(ax_row: np.ndarray, feature: pd.Series, label_maps: pd.DataFrame):
    bin_edges = np.histogram_bin_edges(feature, bins='auto')
    print(f'Plotting histograms for feature {feature.name} with {bin_edges.shape[0]-1} bins.')
    total_hist_log: plt.Axes = ax_row[0]
    total_hist_log.hist(feature, bins=bin_edges, log=True, color='grey')
    total_hist_log.set_title(f'{feature.name} log-space Histogram')
    flood_no_flood = feature[~label_maps[-1]]
    flood_no_flood_hist: plt.Axes = ax_row[1]
    flood_no_flood_hist.hist(flood_no_flood, bins=bin_edges, log=True, color='turquoise')
    flood_no_flood_hist.set_title(f'{feature.name} Histogram without {LABEL_NAMES[-1]}')
    combined_hist: plt.Axes = ax_row[-2]
    combined_flood_hist: plt.Axes = ax_row[-1]
    # width = 0.2
    data_collected = []
    for i, ax, (label, label_name), c in zip(range(3), ax_row[2: -2], LABEL_NAMES.items(), ['red', 'green', 'blue']):
        data = feature[label_maps[label]]
        ax.hist(data, bins=bin_edges, log=True, color=c)
        ax.set_title(f'{label_name} Histogram of {feature.name}')
        data_collected.append(data)
        # combined_hist.bar(np.arange(bin_edges.shape[0]-1)*width*3 + width * (i-1),
        #                   height=np.histogram(data, bins=bin_edges), width=width,
        #                   color=c, log=True)
        # if i > 0:
        #     combined_hist.bar(np.arange(bin_edges.shape[0] - 1)*width*2 - width/2 + (i-1)*width,
        #                       height=np.histogram(data, bins=bin_edges), width=width,
        #                       color=c, log=True)

    combined_hist.hist(data_collected, bins=bin_edges, log=True, color=['red', 'green', 'blue'], histtype='bar',
                       alpha=0.5)
    combined_hist.set_title(f'Comparison of {feature.name} Histograms')
    #combined_hist: plt.Axes = ax_row[-1]
    combined_flood_hist.hist(data_collected[1:], bins=bin_edges, log=True, color=['green', 'blue'], histtype='bar',
                             alpha=0.5)
    combined_flood_hist.set_title(f'Comparison of {feature.name} Histograms')


def plot_1d_histograms(save_folder: str, df: pd.DataFrame, show: bool):
    print(f'Plotting histograms for columns {[c for c in df.columns.values if c != KEY_LABELS]}')
    label_maps = create_label_maps(df)
    num_diagrams = (len(df.columns.values) - 1)
    fig, axs, n_rows = create_figure_with_subplots_capped(f'1D Histograms', num_diagrams, 7)
    save_name = path.join(save_folder, 'histograms')
    n_parts = 1
    filtered_columns = [c for c in df.columns.values if c!= KEY_LABELS]
    for i, column in enumerate(filtered_columns):
        if i > 0 and i % MAX_ROWS == 0:
            name = save_name+f'_{n_parts}' if MAX_ROWS > 1 else save_name+f'_{column}'
            save_and_show_figure(fig, name, show, n_rows)
            fig, axs, n_rows = create_figure_with_subplots_capped(f'1D Histograms',
                                                                  num_diagrams - i,
                                                                  7)
            n_parts += 1
        plot_1d_histogram(axs[i % MAX_ROWS], df[column], label_maps)

    name = save_name + f'_{n_parts}' if MAX_ROWS > 1 else save_name + f'_{filtered_columns[-1]}'
    save_and_show_figure(fig, name, show, n_rows)

def plot_1d_features(save_folder: str, df: pd.DataFrame, show: bool, include_analysis: bool):
    ensure_is_dir(save_folder)
    perform_1d_group_analysis(save_folder, df, show, include_analysis)
    perform_1d_bivariate_analysis(save_folder, df, include_analysis)
    plot_1d_histograms(save_folder,  df, show)

def plot_feature_space(folder: str, f_space_name:str, label_data: ArrayInMemoryDataset, label_meta: Meta,
                       feature_data: ArrayInMemoryDataset, feature_meta: Meta, show: bool = False,
                       include_analysis: bool = False,
                       heatmap_bins: int = 100):
    features: np.ndarray = feature_data.data
    labels: np.ndarray = label_data.data
    labels = labels.reshape(-1)
    features = np.transpose(features, (1, 0) + tuple(range(2, features.ndim)))
    features = features.reshape((features.shape[0], labels.shape[0]))
    df = pd.DataFrame({cn: feature for cn, feature in zip(feature_meta.channel_names, features)})
    df[KEY_LABELS] = pd.Series(data=labels, name=label_meta.channel_names[0])
    del features
    del labels
    plot_1d_features(path.join(folder, '1d_features'), df, show, include_analysis)
    plot_2d_combinations(path.join(folder, '2d_combinations'), df, show, heatmap_bins)


def source_to_dict(source: Pipeline) -> Dict[str, np.ndarray]:
    dataset, meta = source(None, None)
    data: np.ndarray = dataset.data
    data = np.transpose(data, (1, 0) + tuple(range(2, data.ndim)))
    data = data.reshape(data.shape[0], -1)
    return {cn: data_column for cn, data_column in zip(meta.channel_names, data)}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_folder')
    parser.add_argument('result_folder')
    parser.add_argument('-hb', '--heatmap_bins', default=100)
    parser.add_argument('--sci-analysis', dest='include_analysis', action='store_true')
    parser.add_argument('--no-sci-analysis', dest='include_analysis', action='store_false')
    parser.add_argument('--show', dest='show', action='store_true')
    parser.add_argument('--hide', dest='show', action='store_false')
    parser.add_argument('--calc_1d', dest='calc_1d', action='store_true')
    parser.add_argument('--omit_1d', dest='calc_1d', action='store_false')
    parser.add_argument('--calc_2d', dest='calc_2d', action='store_true')
    parser.add_argument('--omit_2d', dest='calc_2d', action='store_false')
    parser.set_defaults(include_analysis=False, show=False, calc_1d=True, calc_2d=True)
    parser.set_defaults()
    args = parser.parse_args()
    result_folder = args.result_folder
    ensure_is_dir(result_folder)
    label_source = Sen1Floods11DataDingsDatasource(args.data_folder, TYPE_LABEL, SPLIT_TRAIN, as_array=True)
    # constructors = construct_feature_spaces(args.data_folder, SPLIT_TRAIN, add_optical=False, as_array=True,
    #                                         allow_lin_ratio=True, allow_sar_hsv=True, filter_method='lee_improved')
    # df = None
    # for f_space, constructor in constructors:
    #     print(f'Viewing {f_space}')
    #     feature_data, feature_meta = constructor(None, None)
    #     assert isinstance(label_data, ArrayInMemoryDataset) and isinstance(feature_data, ArrayInMemoryDataset)
    #     # plot_feature_space(result_folder, f_space, label_data, label_meta, feature_data, feature_meta, show=args.show,
    #     #                    include_analysis=args.include_analysis, heatmap_bins=args.heatmap_bins)
    #     data = np.transpose(feature_data.data, (1, 0) + tuple(range(2, feature_data.data.ndim)))
    #     data = data.reshape((data.shape[0], -1))
    #     if df is None:
    #         df = pd.DataFrame({cn: cn_data for cn, cn_data in zip(feature_meta.channel_names, data)})
    #     else:
    #         for cn, cn_data in zip(feature_meta.channel_names, data):
    #             if cn not in df:
    #                 df[cn] = cn_data
    #             else:
    #                 print(f'Duplicate channel {cn}...')
    s1_source = SequenceModule([
        Sen1Floods11DataDingsDatasource(args.data_folder, TYPE_S1, SPLIT_TRAIN, as_array=True),
        SARFilterModule(method='lee_improved'),
        DistributorModule([
            VV_VH_LinearRatioExtractor()
        ], keep_source=True),
        DistributorModule([
            RangeClippingNormalizationModule('range.json'),
            HSVExtractor('hsv', channels=('VV', 'VV-VH lin. Ratio', 'VH'))
        ], keep_source=True),
        StandardizationModule(standard_file='standard.json', filter_method='lee_improved'),
        InMemoryModule()
    ])
    s2_source = SequenceModule([
        Sen1Floods11DataDingsDatasource(args.data_folder, TYPE_S2, SPLIT_TRAIN, as_array=True),
        DistributorModule([
            AWEIExtractorModule(), AWEISHExtractorModule(), NDWIExtractor(), MNDWI1Extractor(), NDVIExtractor(),
            SequenceModule([
                 WhitelistModule(['SWIR-2', 'NIR', 'Red']),
                 RangeClippingNormalizationModule('range.json'),
                 HSVExtractor('hsv', channels=('SWIR-2', 'NIR', 'Red'))
            ]),
            SequenceModule([
                pl.WhitelistModule(['Red', 'Green', 'Blue']),
                RangeClippingNormalizationModule('range.json'),
                HSVExtractor('hsv')
            ])
        ]),
        StandardizationModule(standard_file='standard.json'),
        InMemoryModule()
    ])
    df = {}
    for i, source in enumerate([s1_source, s2_source, label_source]):#
        print(f'Extracting source number {i}')
        d = source_to_dict(source)
        print(f'Converted to dict with keys {d.keys()}')
        if i == 2:
            df[KEY_LABELS] = d['label']
        elif i == 0:
            df.update({k+' (filtered)':v for k, v in d.items()})
        else:
            df.update(d)
        print(f'New columns are {df.keys()}')
    del s1_source
    del s2_source
    del label_source
    df = pd.DataFrame(df)
    gc.collect()
    print(f'Performing plots for columns {df.columns}')
    if args.calc_1d:
        plot_1d_features(path.join(args.result_folder, '1d_features'), df, show=args.show, include_analysis=args.include_analysis)
    if args.calc_2d:
        plot_2d_combinations(path.join(args.result_folder, '2d_combinations'), df, args.show, args.heatmap_bins)
    labels = df[KEY_LABELS]
    del df
    s2_source = SequenceModule([
        Sen1Floods11DataDingsDatasource(args.data_folder, TYPE_S2, SPLIT_TRAIN, as_array=True),
        StandardizationModule(),
        InMemoryModule()
    ])
    df = pd.DataFrame(source_to_dict(s2_source))
    df[KEY_LABELS] = labels
    del s2_source
    if args.calc_1d:
        plot_1d_features(path.join(args.result_folder, '1d_features'), df, show=args.show,
                         include_analysis=args.include_analysis)
    if args.calc_2d:
        plot_2d_combinations(path.join(args.result_folder, '2d_combinations'), df, args.show, args.heatmap_bins)
