import itertools as itert
import sys
from typing import Optional, Tuple, List, Dict, Set, Any, Type, Union

import faiss
import numba
import numpy as np
import skimage.measure
import skimage.measure as measure
import skimage.segmentation as seg
import sklearn
import sklearn.cluster as cluster
import torch
from torch.utils import data as data

import multipipeline as mpl
import pipeline as pl
import serialisation
import utils


class _ClusteringWrapper(pl.TransformerDataset):
    def __init__(self, wrapped: data.Dataset, kwargs: Optional[dict] = None):
        super().__init__(wrapped)
        self.kwargs = {} if kwargs is None else kwargs

    def _transform(self, data):
        return data.detach().cpu().numpy() if isinstance(data, torch.Tensor) else data

    @staticmethod
    def get_all_kwargs() -> Dict[str, Any]:
        raise NotImplementedError


class _QuickshiftWrapper(_ClusteringWrapper):
    def __init__(self, wrapped: data.Dataset, kwargs: Optional[dict] = None):
        super().__init__(wrapped, kwargs)
        if 'convert2lab' not in self.kwargs:
            self.kwargs['convert2lab'] = False
        if 'kernel_size' not in self.kwargs:
            self.kwargs['kernel_size'] = 1.0
        if 'max_dist' not in self.kwargs:
            self.kwargs['max_dist'] = 4
        if 'ratio' not in self.kwargs:
            self.kwargs['ratio'] = 1.0

    def _transform(self, data):
        data = super()._transform(data)
        # skimage quickshift assumes (width, height, channels) however we have (channel, height, width) (=> transpose)
        # further more it needs the array as a double array (otherwise an error is raised)
        return utils.revert_dims(seg.quickshift(utils.revert_dims(data).astype('double'), **self.kwargs), prefix=(-1,))

    @staticmethod
    def get_all_kwargs() -> Dict[str, Any]:
        return serialisation.get_function_arguments(seg.quickshift, {'image'})


class _ConnectivityLabelingWrapper(_ClusteringWrapper):
    def __init__(self, wrapped: data.Dataset, kwargs: Optional[dict] = None):
        super().__init__(wrapped, kwargs)
        if 'connectivity' not in self.kwargs:
            self.kwargs['connectivity'] = 1

    def _transform(self, data):
        data = super()._transform(data)
        old_shape = data.shape
        data = np.squeeze(data).astype(np.int, copy=False)
        # avoid zero background value
        offset = -np.min(data) + 1
        data += offset
        # now calculate connectivities
        res = measure.label(utils.revert_dims(data), **self.kwargs)
        return utils.revert_dims(res).reshape(old_shape)

    @staticmethod
    def get_all_kwargs() -> Dict[str, Any]:
        return serialisation.get_function_arguments(measure.label, {'input'})


class _PerImageLabelNormalizer(_ClusteringWrapper):
    def __init__(self, wrapped: data.Dataset, kwargs: Optional[dict] = None):
        super().__init__(wrapped, kwargs)
        self.allow_region_props_calc = kwargs.get('allow_region_props_calc', True)

    @staticmethod
    def get_all_kwargs() -> Dict[str, Any]:
        return {'allow_region_props_calc': True}

    def _transform(self, data):
        # data = super()._transform(data)
        # shape = data.shape
        # data = np.squeeze(data)
        # unique_values = np.unique(data)
        # assert len(unique_values.shape) == 1
        # for i, u in enumerate(unique_values):
        #     data[data == u] = i
        # #res = data.reshape(shape)
        # return data.reshape(shape)#res

        data = super()._transform(data)
        shape = data.shape
        data = np.squeeze(data)
        # unique_values = np.unique(data)
        # assert len(unique_values.shape) == 1
        if self.allow_region_props_calc and data.ndim == 2:
            data += -np.min(data) + 1
            for i, p in enumerate(skimage.measure.regionprops(data)):
                bb = np.array(p.bbox).reshape((2, 2))
                indexed = utils.index_by_bounding_box_nonumba(data, bb)
                indexed[indexed == p.label] = i
                utils.assign_in_bounding_box(data, indexed, bb)
        else:
            unique_values = np.unique(data)
            for i, u in enumerate(unique_values):
                data[data == u] = i
        return data.reshape(shape)  # res


def get_constructor(method: str) -> Type[_ClusteringWrapper]:
    method = method.lower().strip()
    if method == 'quickshift':
        return _QuickshiftWrapper
    elif method == 'connectivity_labeling' or method == 'connectivity':
        return _ConnectivityLabelingWrapper
    elif method == 'label_normalize' or method == 'normalize':
        return _PerImageLabelNormalizer
    else:
        raise ValueError('Unknown Method ' + method)


class PerImageClusteringModule(pl.TransformerModule):
    def __init__(self, method: str = 'quickshift', out_channel: Optional[str] = None, kwargs: Optional[dict] = None):
        super().__init__()
        self.method = method
        self.wrapper_cons = get_constructor(method)
        self.out_channel = type(self).__name__[:-8] if out_channel is None else out_channel
        self.kwargs = kwargs or {}

    def get_params_as_dict(self) -> Dict[str, Any]:
        def_params = serialisation.get_function_arguments(self.wrapper_cons.__init__, {'wrapped', 'self'})
        def_params.update(self.kwargs)
        def_params['kwargs'] = self.wrapper_cons.get_all_kwargs()
        def_params['kwargs'].update(self.kwargs)
        return {
            'method': self.method,
            'params': def_params,
            'out_channel': self.out_channel
        }

    def __call__(self, dataset: Optional[data.Dataset], meta: Optional[pl.Meta]) -> Tuple[data.Dataset, pl.Meta]:
        self.print(meta, 'Creating PerImageClustering wrapper.')
        wrapper = self.wrapper_cons(dataset, self.kwargs)
        meta = meta._replace(channel_names=[self.out_channel])
        self.print(meta, f'Successfully created wrapper {type(wrapper).__name__} which outputs to {self.out_channel}.')
        return wrapper, meta


class _PerImageClusteringRefinementWrapper(mpl.MultiTransformerDataset):
    def __init__(self, wrapped: data.Dataset,
                 intensity_dataset: data.Dataset,
                 connectivity: int = 1,
                 perim_area_threshold: float = 12.,
                 convergence_threshold: int = 10,
                 rel_std_threshold: float = 1.,
                 greedy_merge_strategy=True,
                 random_state: np.random.Generator = None,
                 max_iter: int = 10,
                 max_merge_trials: int = 100,
                 cache: bool = True,
                 remove_singular: bool = True):
        super().__init__(wrapped)
        assert 1 <= connectivity <= 2
        self.intensity_dataset = intensity_dataset
        self.cache = cache
        self.index_array = None
        self.connectivity = connectivity
        self.perim_area_threshold = perim_area_threshold
        self.convergence_threshold = convergence_threshold
        self.rel_std_threshold = rel_std_threshold
        self.greedy_merge_strategy = greedy_merge_strategy
        self.random_state: np.random.Generator = np.random.default_rng(42) if random_state is None else random_state
        self.max_iter: int = max_iter
        self.max_merge_trials: int = max_merge_trials
        self.remove_singular = remove_singular

    def _get_sorted_slice_arrays(self, label_image: np.ndarray, intensity_image: np.ndarray) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        assert label_image.shape[0] == label_image.shape[1]
        # if (self.index_array is None) or (self.index_array.shape[0] != label_image.shape[0]):
        #    self.index_array = np.arange(label_image.shape[0])
        # ia = self.index_array
        # ia_max = self.index_array[-1]

        props: List[skimage.measure._regionprops.RegionProperties] = \
            measure.regionprops(label_image, intensity_image=intensity_image,
                                cache=self.cache, extra_properties=[utils.property_function('std')])
        # first of all, create an easily extendable index array out of the slices
        # bb_slices = [[ia[sl] for sl in p.slice if sl is not None] for p in props]
        # now extend these indices by 1 in both directions in order to also capture all neighbours
        # also wrap it in an array, so that we can sort it by label
        # bb_slices = np.array([[ia[max(sl[0]-1, 0):min(sl[-1]+1, ia_max)+1] for sl in t]
        #                       for t in bb_slices], dtype=np.object)
        # now the remaining properties
        mean = np.array([prop.mean_intensity for prop in props])
        std = np.array([prop.standard_deviation for prop in props])
        bbs = np.array([prop.bbox for prop in props]).reshape((-1, 2, 2))
        areas = np.array([prop.area for prop in props]) if self.remove_singular else None
        # extract labels and a sort array
        labels = np.array([p.label for p in props])
        label_sort = np.argsort(labels)
        # sort them...
        # bb_slices = bb_slices[label_sort]
        labels = labels[label_sort]
        mean = mean[label_sort]
        std = std[label_sort]
        bbs = bbs[label_sort]
        areas = [None] * len(bbs) if areas is None else areas[label_sort]
        # indexing idiom: [(label_image[sl[0]])[:,sl[1]] == labels[i] for i, sl in enumerate(bb_slices)]
        return labels, mean, std, bbs, areas

    @staticmethod
    @numba.njit()
    def _get_neighbours(label_image: np.ndarray, label: int, bbs: np.ndarray, connectivity: int) -> np.ndarray:
        max_bb_val = label_image.shape[0]  # not minus 1!!! This is exclusive!!!
        expanded_bbs = np.array([[max(bbs[0, 0] - 1, 0), max(bbs[0, 1] - 1, 0)],
                                 [min(bbs[1, 0] + 1, max_bb_val), min(bbs[1, 1] + 1, max_bb_val)]])
        img_slice = utils.index_by_bounding_box(label_image, expanded_bbs)
        sliced_mask = img_slice == label
        # shift in all 1-connectivity directions to get the neighbours
        shift_first = np.logical_or(utils.shift_1(sliced_mask, 1, False), utils.shift_1(sliced_mask, -1, False))
        shift_2_1 = utils.shift_2(sliced_mask, 1, False)
        shift_2_2 = utils.shift_2(sliced_mask, -1, False)
        shift_second = np.logical_or(shift_2_1, shift_2_2)
        combined = np.logical_or(shift_first, shift_second)
        # if we now also need the 2 connectivity, shift it up and down as well
        if connectivity == 2:
            shift_first = np.logical_or(utils.shift_1(shift_2_1, 1, False), utils.shift_1(shift_2_1, -1, False))
            shift_second = np.logical_or(utils.shift_1(shift_2_2, 1, False), utils.shift_1(shift_2_2, -1, False))
            combined = np.logical_or(combined, np.logical_or(shift_first, shift_second))
        # now get the unique values in the result area
        # these will be all possible neighbours as well as the label (likely slower to do another mask to remove
        # the label, much simpler to remove it from the result set)
        # notice that flatten won't change the result here as that's what unique is going to do anyway
        # without this numba won't work as it doesn't know about this equivalence
        return np.unique(img_slice.flatten()[combined.flatten()])

    @staticmethod
    def _get_neighbours_set(label_image: np.ndarray, label: int, bbs: np.ndarray, connectivity: int) -> Set[int]:
        res = set(_PerImageClusteringRefinementWrapper._get_neighbours(label_image, label, bbs, connectivity))
        # and of the course the label isn't a neighbour of itself...
        if label in res:
            res.remove(label)
        return res

    def _extract_neighbours_and_info(self, label_image: np.ndarray, intensity_image: np.ndarray) \
            -> Dict[int, Tuple[Set[int], np.ndarray, np.ndarray, np.ndarray]]:
        """

        :param label_image: the label-image to process. should be of integer type
        :param intensity_image: the intensity image to process, should be of float type
        :return: a dictionary containing all labels as keys and a tuple with neighbours, mean, std and bounding box
        array as the value. Will be sorted by label index.
        """

        sorted_slice_arrays = self._get_sorted_slice_arrays(label_image, intensity_image)
        return {label: (self._get_neighbours_set(label_image, label, bbs, self.connectivity), mean, std, bbs, areas)
                for label, mean, std, bbs, areas in zip(*sorted_slice_arrays)}

    def _try_merge_with_shape_constraint(self, intensity_image, label_image, per_label_info, to_merge, excluded,
                                         label, nb_set, bbs, possible_neighbours) -> bool:
        bb = utils.bbs_union(itert.chain.from_iterable([[bbs],
                                                        [per_label_info[nb][3]
                                                         for nb in possible_neighbours]]))
        indexed_label_image = np.copy(utils.index_by_bounding_box(label_image, bb))
        for nb in possible_neighbours:
            indexed_label_image[indexed_label_image == nb] = label
        merged_prop = next(filter(lambda p: p.label == label,
                                  measure.regionprops(indexed_label_image,
                                                      intensity_image=utils.index_by_bounding_box(
                                                          intensity_image, bb),
                                                      extra_properties=[utils.property_function('std')],
                                                      cache=self.cache)))
        new_area = merged_prop.area
        if merged_prop.perimeter / np.sqrt(new_area) >= self.perim_area_threshold:
            return False
        res_nb_set = nb_set.copy()
        for nb in possible_neighbours:
            res_nb_set.update(per_label_info[nb][0])
        res_tup = res_nb_set, merged_prop.mean_intensity, merged_prop.standard_deviation, bb, new_area
        # this is safe as we have excluded all neighbours from further investigation
        utils.assign_in_bounding_box(label_image, indexed_label_image, bb)
        excluded.update(possible_neighbours)
        del indexed_label_image
        possible_neighbours = set(possible_neighbours)
        to_merge[label] = (possible_neighbours, res_tup)
        excluded.update(possible_neighbours)
        return True

    def _try_merge_non_greedy(self, intensity_image, label_image, per_label_info, to_merge, excluded,
                              label, nb_set, bbs, possible_neighbours):
        merged_successfully = False
        neighbour_array = np.array(possible_neighbours)
        i = 0
        tested_samples: Set[Tuple[int, ...]] = set()
        while not merged_successfully and i < self.max_merge_trials:
            num_neighbours_to_try = self.random_state.integers(1, len(possible_neighbours), 1, endpoint=True)
            neighbours_to_use = self.random_state.choice(neighbour_array, num_neighbours_to_try, replace=False)
            ntu_tup = tuple(neighbours_to_use)
            if ntu_tup not in tested_samples:
                merged_successfully = self._try_merge_with_shape_constraint(intensity_image, label_image,
                                                                            per_label_info, to_merge, excluded,
                                                                            label, nb_set, bbs, neighbours_to_use)
                tested_samples.add(ntu_tup)
            i += 1

    def _try_merge(self, intensity_image, label_image, per_label_info, to_merge, excluded):
        for label, (nb_set, mean, std, bbs, _) in per_label_info.items():
            # first check if any neighbours are left that are not yet going to be merged
            possible_neighbours = [nb for nb in nb_set if
                                   (nb not in to_merge) and (nb not in excluded)]  # and nb in per_label_info]
            if not possible_neighbours:
                continue
            min_std = [np.min(np.stack((std, (per_label_info[nb])[2])), axis=0) * self.rel_std_threshold for nb in
                       possible_neighbours]
            possible_neighbours = [nb for i, nb in enumerate(possible_neighbours)
                                   if np.all(np.abs(mean - per_label_info[nb][1]) < min_std[i])]
            if not possible_neighbours:
                continue
            # res_tup = None, None, None, None
            if self.greedy_merge_strategy:
                self._try_merge_with_shape_constraint(intensity_image, label_image, per_label_info, to_merge, excluded,
                                                      label, nb_set, bbs, possible_neighbours)
            else:
                self._try_merge_non_greedy(intensity_image, label_image, per_label_info, to_merge, excluded, label,
                                           nb_set, bbs, possible_neighbours)

    def _merge_singular(self, intensity_image, label_image, per_label_info, to_merge, excluded):
        for label, (nb_set, mean, std, bbs, area) in per_label_info.items():
            if area > 1:
                continue
            # first check if any neighbours are left that are not yet going to be merged and are non-singular
            possible_neighbours = [(nb, per_label_info[nb][1]) for nb in nb_set if
                                   (nb not in to_merge) and (nb not in excluded) and per_label_info[nb][-1] > 1]
            if not possible_neighbours:
                continue
            # landuyt et al. merge singular segments with the segment of least possible mean difference...
            # this is in contrast to a normal merge, as that would also utilise the std threshold to prevent merging
            # if it falls "out of distribution"
            best_mean_neighbour_i = np.argmin(
                np.sum(np.abs(np.array([mean for _, mean in possible_neighbours]) - mean), axis=1))
            nb, _ = possible_neighbours[best_mean_neighbour_i]
            self._try_merge_with_shape_constraint(intensity_image, label_image, per_label_info, to_merge, excluded,
                                                  label, nb_set, bbs, [nb])

    def _transform(self, data, index):
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        label_image = utils.revert_dims(data)
        reverted_shape = label_image.shape
        label_image = np.squeeze(label_image).astype(np.int)
        intensity_image = utils.revert_dims(self.intensity_dataset[index])
        # t = time.time()
        per_label_info = self._extract_neighbours_and_info(label_image, intensity_image)
        # print(f'Neighbour extraction took {time.time()-t:.3f}s. Performing merge on input with {len(per_label_info)} clusters.')
        # t = time.time()
        last_merges = 0
        total_merges = 0
        total_eliminated = 0
        i = -1 if self.remove_singular else 0
        while i <= 0 or (i < self.max_iter and last_merges > self.convergence_threshold):
            # a dict containing (in both directions!!!) which segments to merge
            to_merge: Dict[int, Tuple[Set[int], Tuple[Set[int], np.ndarray, np.ndarray, np.ndarray]]] = {}
            excluded: Set[int] = set()
            if i >= 0:
                self._try_merge(intensity_image, label_image, per_label_info, to_merge, excluded)
            else:
                self._merge_singular(intensity_image, label_image, per_label_info, to_merge, excluded)
                # print(f'Merging {label} with {str(possible_neighbours)}.')
            # print('Completed merge-search. Fixing per_label_info')
            last_merges = len(to_merge)
            total_merges += last_merges
            total_eliminated += len(excluded)
            while to_merge:
                next_to_merge, (neighbours, res_tup) = next(iter(to_merge.items()))
                per_label_info[next_to_merge] = res_tup
                to_merge.pop(next_to_merge, None)
            per_label_info = {label: (neighbours_to_clear.difference(excluded), mean, std, bb, area)
                              for label, (neighbours_to_clear, mean, std, bb, area) in per_label_info.items()
                              if label not in excluded}
            i += 1
            # print(f'Completed merge-iteration with {last_merges} performed merges. Removed {len(excluded)} clusters, {len(per_label_info)} remaining.')
        # print(f'Performed {total_merges} merges which took {time.time()-t:.3f}s and eliminated {total_eliminated} clusters.')
        return utils.revert_dims(label_image.reshape(reverted_shape))


class PerImageClusteringRefinementModule(mpl.MultiTransformerModule):
    def __init__(self, label_criterion: mpl.SelectionCriterion,
                 intensity_criterion: mpl.SelectionCriterion,
                 connectivity: int = 1,
                 perim_area_threshold: float = 12.,
                 convergence_threshold: int = 10,
                 rel_std_threshold: float = 1.,
                 greedy_merge_strategy=True,
                 random_state: int = 42,
                 max_iter: int = 10,
                 max_merge_trials: int = 100,
                 result_name: Optional[str] = None,
                 cache: bool = True,
                 remove_singular: bool = True):
        super().__init__()
        self.label_criterion = label_criterion
        self.intensity_criterion = intensity_criterion
        self.cache = cache
        self.result_name = result_name
        self.connectivity: int = connectivity
        self.perim_area_threshold = perim_area_threshold
        self.convergence_threshold = convergence_threshold
        self.rel_std_threshold = rel_std_threshold
        self.greedy_merge_strategy = greedy_merge_strategy
        self.random_state: int = random_state
        self.max_iter: int = max_iter
        self.max_merge_trials: int = max_merge_trials
        self.remove_singular = remove_singular

    def __call__(self, summary: mpl.Summary) -> mpl.Summary:
        self.print(summary, 'Generating wrapper to perform per-image segmentation refinement using label criterion '
                            f'{self.label_criterion} and intensity index {self.intensity_criterion}.')
        label_name, (label_dataset, label_meta) = summary.by_criterion(self.label_criterion,
                                                                       delete=True,
                                                                       return_name=True)
        intensity_dataset, intensity_meta = summary.by_criterion(self.intensity_criterion)
        if not isinstance(label_dataset, pl.ShapelessInMemoryDataset):
            self.print(summary, 'WARNING: Labels are not in-memory. This may lead to poor performance!')
        if not isinstance(intensity_dataset, pl.ShapelessInMemoryDataset):
            self.print(summary, 'WARNING: Intensities are not in-memory. This may lead to poor performance!')
        label_meta: pl.Meta = label_meta
        rand = np.random.default_rng(
            self.random_state) if label_meta.run_hook is None else label_meta.run_hook.numpy_rng
        label_dataset = _PerImageClusteringRefinementWrapper(label_dataset, intensity_dataset, self.connectivity,
                                                             self.perim_area_threshold, self.convergence_threshold,
                                                             self.rel_std_threshold, self.greedy_merge_strategy,
                                                             rand, self.max_iter, self.max_merge_trials,
                                                             self.cache, self.remove_singular)
        self.print(summary,
                   f'Successfully created refinement wrapper featuring intensity channels {str(intensity_meta.channel_names)}.')
        return summary.add(label_dataset, label_meta, label_name if self.result_name is None else self.result_name)


class FaissKMeans(sklearn.base.ClassifierMixin, sklearn.base.BaseEstimator):
    def __init__(self, k: int = 5, spherical: bool = False, max_iter: int = 20, nredo: int = 3,
                 n_points_per_cluster: Optional[Union[int, float]] = None, random_state: int = 42,
                 use_gpu: bool = True, verbose: bool = False) -> None:
        super().__init__()
        self.k = k
        self.spherical = spherical
        self.use_gpu = use_gpu
        self.max_iter = max_iter
        self.nredo = nredo
        self.n_points_per_cluster = n_points_per_cluster
        self.random_state = random_state
        self.verbose = verbose
        self.centroids = None
        self.inertia_ = None

    def fit(self, X, y=None):
        if self.use_gpu:
            gpu_resource = faiss.StandardGpuResources()
            index = faiss.GpuIndexFlatL2(gpu_resource, X.shape[1])
        else:
            index = faiss.IndexFlatL2(X.shape[1])
        if self.n_points_per_cluster is None:
            max_points_per_centroid = X.shape[0] // self.k
        elif type(self.n_points_per_cluster) is int:
            max_points_per_centroid = self.n_points_per_cluster
        else:
            max_points_per_centroid = int(self.n_points_per_cluster * X.shape[0])
        # faiss_k_means = faiss.Clustering(X.shape[1], self.k, niter=self.max_iter, nredo=self.nredo,seed=self.random_state,
        #                              max_points_per_centroid=max_points_per_centroid, spherical=self.spherical,
        #                              gpu=self.use_gpu)
        X = X.astype(np.float32)
        if not X.flags['C_CONTIGUOUS']:
            X = np.ascontiguousarray(X)
        faiss_k_means = faiss.Clustering(X.shape[1], self.k)
        faiss_k_means.niter = self.max_iter
        faiss_k_means.nredo = self.nredo
        faiss_k_means.seed = self.random_state
        faiss_k_means.max_points_per_centroid = max_points_per_centroid
        faiss_k_means.spherical = self.spherical
        faiss_k_means.verbose = self.verbose

        faiss_k_means.train(X, index)
        self.centroids = faiss.vector_to_array(faiss_k_means.centroids).reshape((self.k, -1))

    def predict(self, X):
        X = X.astype(np.float32)
        if not X.flags['C_CONTIGUOUS']:
            X = np.ascontiguousarray(X)
        if self.use_gpu:
            gpu_resource = faiss.StandardGpuResources()
            index = faiss.GpuIndexFlatL2(gpu_resource, self.centroids.shape[1])
        else:
            index = faiss.IndexFlatL2(self.centroids.shape[1])
        index.add(self.centroids)
        _, I = index.search(X, k=1)
        return I.reshape(-1)


class SerializableDBScan(cluster.DBSCAN, serialisation.JSONSerializable):
    def __init__(self, eps=0.5, *, min_samples=5, metric="euclidean", metric_params=None, algorithm="auto",
                 leaf_size=30, p=None, n_jobs=None):
        super().__init__(eps, min_samples=min_samples, metric=metric, metric_params=metric_params, algorithm=algorithm,
                         leaf_size=leaf_size, p=p, n_jobs=n_jobs)

    def predict(self, X=None, y=None, sample_weight=None):
        if sample_weight is not None and isinstance(sample_weight, np.ndarray) and sample_weight.ndim > 1 and not any(map(lambda s: s> 1, sample_weight.shape[1:])):
            sample_weight = sample_weight.flatten()
        res = self.fit_predict(X,y,sample_weight)
        return res

    def get_params_as_dict(self) -> Dict[str, Any]:
        return self.get_params()


class SerializableAgglomerative(cluster.AgglomerativeClustering, serialisation.JSONSerializable):
    def __init__(self, n_clusters=2, *, affinity="euclidean", memory=None, connectivity=None, compute_full_tree="auto",
                 linkage="ward", distance_threshold=None, compute_distances=False):
        super().__init__(n_clusters, affinity=affinity, memory=memory, connectivity=connectivity,
                         compute_full_tree=compute_full_tree, linkage=linkage, distance_threshold=distance_threshold,
                         compute_distances=compute_distances)

    def fit_predict(self, X, y=None):
        if X.shape[0] <= self.n_clusters:
            print(f'Given {X.shape[0]} samples but asked for {self.n_clusters}. This is impossible, therefore skipping!',
                  file=sys.stderr)
            return X
        return super().fit_predict(X, y)

    def predict(self, X=None, y=None):
        res = self.fit_predict(X,y)
        return res

    def get_params_as_dict(self) -> Dict[str, Any]:
        return self.get_params()

class SerializableDirectBirch(cluster.Birch, serialisation.JSONSerializable):
    def predict(self, X=None, y=None):
        return self.fit_predict(X,y)

    def get_params_as_dict(self) -> Dict[str, Any]:
        return self.get_params()


class MeanShiftWithRobustSeeding(cluster.MeanShift, serialisation.JSONSerializable):
    def fit(self, X, y=None):
        prev_seeds = self.seeds
        if self.bin_seeding and self.seeds is None:
            self.seeds: np.ndarray = cluster.get_bin_seeds(X, self.bandwidth, self.min_bin_freq)
            halve_bin_freq = True
            while self.seeds.shape[0] == 0:
                print(f'No seeds could be calculated with bandwidth={self.bandwidth} and '
                      f'min_bin_freq={self.min_bin_freq}!', file=sys.stderr)
                if halve_bin_freq and self.min_bin_freq >= 2:
                    print(f'Halving min_bin_freq to allow more seeds to be valid!', file=sys.stderr)
                    self.min_bin_freq = self.min_bin_freq // 2
                    halve_bin_freq = False
                else:
                    print(f'Doubling bandwidth in order to provide a coarser grid!', file=sys.stderr)
                    self.bandwidth *= 2.0
                    halve_bin_freq = True
                self.seeds: np.ndarray = cluster.get_bin_seeds(X, self.bandwidth, self.min_bin_freq)
        res = super().fit(X, y)
        self.seeds = prev_seeds
        return res

    def get_params_as_dict(self) -> Dict[str, Any]:
        return self.get_params()
try:
    import cuml.cluster
    class DirectPredictRDBSCAN(cuml.cluster.DBSCAN, serialisation.JSONSerializable):
        def predict(self, X):
            return self.fit_predict(X)

        def get_params_as_dict(self) -> Dict[str, Any]:
            params = super().get_params_as_dict()
            valid_params = {"eps",  "min_samples", "max_mbytes_per_batch", "calc_core_sample_indices", "metric"}
            params = {k:p for k,p in params.items() if k in valid_params and not isinstance(p, cuml.Handle)}
            return params

except:
    print(f'Failed to import cuml!', file=sys.stderr)

try:
    import sklearnex.cluster
    class DirectPredictIXDBSCAN(sklearnex.cluster.DBSCAN, serialisation.JSONSerializable):
        def predict(self, X):
            return self.fit_predict(X)

        def get_params_as_dict(self) -> Dict[str, Any]:
            params = super().get_params_as_dict()
            valid_params = {"eps",  "min_samples", "metric"}
            params = {k:p for k,p in params.items() if k in valid_params}
            return params

except:
    print(f'Failed to import sklearnex.cluster!', file=sys.stderr)