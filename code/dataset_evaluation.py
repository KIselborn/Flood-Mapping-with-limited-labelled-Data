import argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

from data_source import *
from pipeline import *
from plot_utils import *

_cmap = cm.get_cmap('tab20c')(np.array([4, 8, 0, 16]))
FLOOD_COLORS = {(i-1):_cmap[i] for i in range(3)}
IMAGE_TYPE_MAP = {
    (True, False, True): 'Unclassified Data',
    (False, True, True): 'Dry Land',
    (False, False, False): 'Relevant Flood Data',
    (False, False, True): 'No Flood Data',
}
IMAGE_TYPE_COLOR_MAP = {k: _cmap[i] for i, k in enumerate(IMAGE_TYPE_MAP.keys())}

IMAGE_TYPE_MAP_ALL = IMAGE_TYPE_MAP.copy()
for perm in itert.combinations([True, False], 3):
    if perm not in IMAGE_TYPE_MAP_ALL:
        IMAGE_TYPE_MAP_ALL[perm] = 'Illegal State'

def plot_distribution(df: pd.DataFrame, flood_images: pd.DataFrame, split: Optional[str], axes: np.ndarray, region_colors: Dict[str, Any]) \
        -> Dict[str, Any]:
    count_axis: plt.Axes = axes[0]
    def percentage_display_total(pct: float):
        return f'{pct:.2f}%'
    # see https://matplotlib.org/stable/gallery/pie_and_polar_charts/nested_pie.html
    # and https://matplotlib.org/stable/gallery/pie_and_polar_charts/pie_demo2.html
    count_dict = df.drop(columns=['region']).groupby('data').sum().to_dict()['count']
    no_data_dry_flood_colors = [FLOOD_COLORS[flood_state] for flood_state in count_dict.keys()]
    vals = np.array(list(count_dict.values())).flatten()
    wedges_outer, texts_outer, autotexts_inner = count_axis.pie(vals,
                                                      radius=1, colors=no_data_dry_flood_colors,
                                                      wedgeprops=dict(edgecolor='w'),
                                                      autopct=percentage_display_total,
                                                      labels=list(FLOOD_LABELS.values())[len(FLOOD_LABELS)-vals.shape[0]:])

    plt.setp(texts_outer, visible=False)
    plt.setp(autotexts_inner, size=12, weight='bold', color='w')
    count_axis.set_title(f'Flood-No-Flood {"" if split is None else split+"-set "}distribution')
    count_axis.legend()


    flood_img_axis: plt.Axes = axes[1]
    flooded_series: pd.Series = flood_images.drop(columns=['region']).groupby(['is_no_data_only','is_dry_only', 'is_no_flood']).size()
    labels = [IMAGE_TYPE_MAP_ALL[label] for label in flooded_series.index.values]
    flood_image_colors = [IMAGE_TYPE_COLOR_MAP[key] for key in flooded_series.index.values]
    flood_no_flood_series = flood_images.drop(columns=['is_no_data_only','is_dry_only', 'region']).groupby(['is_no_flood']).size()
    wedges_outer, _, autotexts_inner = flood_img_axis.pie(
        flood_no_flood_series,
        radius=1, colors=cm.get_cmap('tab20c')(np.array([0, 3])),
        autopct=percentage_display_total,
        wedgeprops=dict(edgecolor='w', width=0.3))#,
        #labels=list(IMAGE_TYPE_MAP.values())[2:] if len(flood_no_flood_series) > 1 else list(IMAGE_TYPE_MAP.values())[2:3])
    wedges_outer, texts = flood_img_axis.pie(
        flooded_series,
        radius=1-0.3, colors=flood_image_colors,
        wedgeprops=dict(edgecolor='w', width=0.45),
        labels=labels)

    plt.setp(autotexts_inner, size=18, weight='bold', color='w')
    plt.setp(texts, size=12, weight='bold', visible=False)
    flood_img_axis.legend()
    flood_img_axis.set_title(f'Percentage of images with floods{"" if split is None else " in "+split+"-set"}')

    per_region: pd.DataFrame = df.drop(columns=['data']).groupby('region').sum().reset_index('region')
    per_region = per_region.sort_values(by=['count', 'region'], ascending=False)
    region_order = per_region['region']
    color_map = cm.get_cmap('Set3')
    for region in sorted(per_region['region']):
        if region not in region_colors:
            region_colors[region] = color_map([len(region_colors)])[0]

    ordered_colors = [region_colors[region] for region in per_region['region']]

    dist_axis: plt.Axes = axes[2]
    x_positions = np.arange(per_region.shape[0])
    total_sum: float = float(per_region['count'].sum())
    dist_axis.bar(x_positions,
                  height=100.0 * per_region['count'] / total_sum,
                  width=0.4, color=ordered_colors)
    dist_axis.set_xticks(x_positions)
    dist_axis.set_xticklabels(region_order, rotation=45, ha='right')
    dist_axis.set_xlabel('Region')
    dist_axis.set_ylabel('Fraction of Dataset in %' if split is None else 'Fraction of Split in %')
    dist_axis.set_title(f'Region distribution{"" if split is None else " in "+split+"-set "}')

    # fsbr = flood_state_by_region
    fsbr_axis: plt.Axes = axes[3]
    width = 0.3
    for flood_state in [-1, 0, 1]:
        region_wise_counts = df.loc[df['data'] == flood_state, ['region', 'count']]
        relevant_counts = [float(region_wise_counts.loc[region_wise_counts['region'] == region, ['count']].sum())/
                           per_region.loc[per_region['region'] == region, ['count']].sum()
                           for region in region_order]
        relevant_counts = np.array(relevant_counts).flatten()
        assert relevant_counts.shape[0] == x_positions.shape[0]
        fsbr_axis.bar(x_positions+width*flood_state,
                      height=100.0 * relevant_counts,
                      width=width, color=FLOOD_COLORS[flood_state],
                      label=FLOOD_LABELS[flood_state])
    fsbr_axis.set_xticks(x_positions)
    fsbr_axis.set_xticklabels(region_order, rotation=45, ha='right')
    fsbr_axis.set_xlabel('Region')
    fsbr_axis.set_ylabel('Fraction of Region in %')
    fsbr_axis.set_title(f'Flood state distribution by region distribution{"" if split is None else " in "+split+"-set"}')
    fsbr_axis.legend()

    fibr_axis: plt.Axes = axes[4]
    width = 0.2
    grouped_by_flooded: pd.DataFrame = flood_images.groupby(['is_no_data_only','is_dry_only', 'is_no_flood', 'region'])\
        .size().reset_index(name='count')
    num_per_region: pd.DataFrame = flood_images.groupby(['region']).size().reset_index(name='count')
    for i, (descriptor, label) in enumerate(IMAGE_TYPE_MAP.items()):
        if not np.any(grouped_by_flooded[['is_no_data_only','is_dry_only', 'is_no_flood']] == descriptor):
            continue
        relevant_counts = [(grouped_by_flooded.loc[np.all(grouped_by_flooded[['is_no_data_only','is_dry_only', 'is_no_flood', 'region']]
                                                    == (descriptor + (region,)), axis=1), 'count']
                            .sum() / num_per_region.loc[num_per_region['region'] == region, 'count'].sum())
                           for region in region_order]
        relevant_counts = np.array(relevant_counts).flatten()
        assert relevant_counts.shape[0] == x_positions.shape[0]
        fibr_axis.bar(x_positions + i*width - 1.5 * width,
                      height=100.0 * relevant_counts,
                      width=width, color=IMAGE_TYPE_COLOR_MAP[descriptor],
                      label=label)
    fibr_axis.set_xticks(x_positions)
    fibr_axis.set_xticklabels(region_order, rotation=45, ha='right')
    fibr_axis.set_xlabel('Region')
    fibr_axis.set_ylabel('Fraction of Region in %')
    fibr_axis.set_title(f'Percentage of images with Floods by region{"" if split is None else " in "+split+"-set"}')
    fibr_axis.legend()
    return region_colors


def plot_set_distribution(split: str,axes: np.ndarray, dataset: data.Dataset, meta: Meta, region_colors: Dict[str, Any]) \
        -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    regions = [info.region for info in meta.per_item_info]
    unique_regions = list(sorted(set(regions)))
    if isinstance(dataset, ArrayInMemoryDataset):
        # Legacy code based on loading the whole thing into memory...
        data = dataset.data
        cat_type = pd.Categorical([region for region in regions for _ in range(data.shape[-2]) for _ in range(data.shape[-1])],
                                  categories=unique_regions, ordered=True)
        df = pd.DataFrame({'data': data.reshape(-1), 'region': pd.Series(cat_type, name='region')})
        df: pd.DataFrame = df.groupby(['region', 'data']).size().reset_index(name='count')
        flood_images: pd.DataFrame = pd.DataFrame({'is_dry_only': [np.all(ar == 0) for ar in data],
                                                   'is_no_data_only': [np.all(ar == -1) for ar in data],
                                                   'is_no_flood': [np.all(ar != 1) for ar in data],
                                                   'region': pd.Series(pd.Categorical(regions, categories=unique_regions, ordered=True))})
    else:
        region_class_tuples = []
        flood_image_tuples = []
        for img, info in zip(dataset, meta.per_item_info):
            per_item_info: PerItemMeta = info
            is_dry_only, is_no_data_only, is_no_flood = True, True, True
            for clazz in [-1, 0, 1]:
                num_with_clazz = np.count_nonzero(img == clazz)
                if num_with_clazz > 0:
                    region_class_tuples.append((per_item_info.region, clazz, num_with_clazz))
                    is_dry_only = is_dry_only and clazz == 0
                    is_no_data_only = is_no_data_only and clazz == -1
                    is_no_flood = is_no_flood and clazz != 1
            flood_image_tuples.append((is_dry_only, is_no_data_only, is_no_flood, per_item_info.region))
        cat_type = pd.Categorical([region for region, _, _ in region_class_tuples], categories=unique_regions,
                                  ordered=True)
        df: pd.DataFrame = pd.DataFrame({'region': pd.Series(cat_type, name='region'),
                                         'data': [clazz for _, clazz, _ in region_class_tuples],
                                         'count': [count for _, _, count in region_class_tuples]})
        cat_type = pd.Categorical([region for _, _, _, region in flood_image_tuples], categories=unique_regions,
                                  ordered=True)
        flood_images: pd.DataFrame = pd.DataFrame({
            'is_dry_only': [is_dry_only for is_dry_only, _, _, _ in flood_image_tuples],
            'is_no_data_only': [is_no_data_only for _, is_no_data_only, _, _ in flood_image_tuples],
            'is_no_flood': [is_no_flood for _, _, is_no_flood, _ in flood_image_tuples],
            'region': pd.Series(cat_type, name='region')
        })

    region_colors = plot_distribution(df, flood_images, split, axes, region_colors)
    return region_colors, df, flood_images


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_folder')
    parser.add_argument('plot_folder')
    parser.add_argument('--sci-analysis', dest='include_analysis', action='store_true')
    parser.add_argument('--no-sci-analysis', dest='include_analysis', action='store_false')
    parser.add_argument('--show', dest='show', action='store_true')
    parser.add_argument('--hide', dest='show', action='store_false')
    parser.add_argument('--save', dest='save', action='store_true')
    parser.add_argument('--no-save', dest='save', action='store_false')
    parser.set_defaults(include_analysis=False, show=True, save=True)
    parser.set_defaults()
    args = parser.parse_args()
    SPLITS_TO_SHOW = [SPLIT_TRAIN, SPLIT_VALIDATION, SPLIT_TEST, SPLIT_BOLIVIA, SPLIT_WEAK]
    fig, axes, ld = create_figure_with_subplots('Sen1Floods11-per-Split Distribution', 5, len(SPLITS_TO_SHOW))
    region_colors = {}
    flood_pixels_by_region, flood_images = [], []
    for i, split in enumerate(SPLITS_TO_SHOW):
        source = Sen1Floods11DataDingsDatasource(args.data_folder, (TYPE_LABEL if split != SPLIT_WEAK else
                                                                    TYPE_S2_WEAK_LABEL),
                                                 split, as_array=False, in_memory=False)
        dataset, meta = source(None, None)
        region_colors, flood_pixels_by_region_df, flood_images_df = plot_set_distribution(split, axes[:, i], dataset, meta, region_colors)
        if split != SPLIT_WEAK:
            flood_pixels_by_region.append(flood_pixels_by_region_df)
            flood_images.append(flood_images_df)
    save_and_show(fig, path.join(args.plot_folder, 'dataset_split_dist'), args.show, ld, args.save)
    fig, axes, ld = create_figure_with_subplots('Sen1Floods11-Hand-Labeled Distribution', 1, 5)
    plot_distribution(pd.concat(flood_pixels_by_region).sort_values(['count', 'region']),
                      pd.concat(flood_images).sort_values(['region', 'is_no_data_only', 'is_dry_only', 'is_no_flood']),
                      None, axes[0], region_colors)
    save_and_show(fig, path.join(args.plot_folder, 'dataset_dist'), args.show, ld, args.save)
