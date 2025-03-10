import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import math
import graph_utils
import rtree
import scipy
import pickle
import os
import addict
import json
import logging
from utils import load_data_config
# Import DatasetHandler for GCP download functionality
from extract.extractors.georeferencers.utils.get_datsets import DatasetHandler



def read_rgb_img(path):
    bgr = cv2.imread(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb

def os_data_partition():
    with open('./os/data_split.json','r') as jf:
        data_list = json.load(jf)
    train_list = data_list['train']
    val_list = data_list['validation']
    test_list = data_list['test']
    return train_list, val_list, test_list

def get_patch_info_one_img(image_index, image_size, sample_margin, patch_size, patches_per_edge):
    patch_info = []
    sample_min = sample_margin
    sample_max = image_size - (patch_size + sample_margin)
    eval_samples = np.linspace(start=sample_min, stop=sample_max, num=patches_per_edge)
    eval_samples = [round(x) for x in eval_samples]
    for x in eval_samples:
        for y in eval_samples:
            patch_info.append(
                (image_index, (x, y), (x + patch_size, y + patch_size))
            )
    return patch_info


class GraphLabelGenerator():
    def __init__(self, config, full_graph, coord_transform):
        self.config = config
        # full_graph: sat2graph format
        # coord_transform: lambda, [N, 2] array -> [N, 2] array
        # convert to igraph for high performance

        self.full_graph_origin = graph_utils.igraph_from_adj_dict(full_graph, coord_transform,dataset=config.DATASET)
        
        # find crossover points, we'll avoid predicting these as keypoints
        self.crossover_points = graph_utils.find_crossover_points(self.full_graph_origin)
        # subdivide version
        # TODO: check proper resolution
        self.subdivide_resolution = 4
        self.full_graph_subdivide = graph_utils.subdivide_graph(self.full_graph_origin, self.subdivide_resolution)
        # np array, maybe faster
        self.subdivide_points = np.array(self.full_graph_subdivide.vs['point'])
        # pre-build spatial index
        # rtree for box queries
        self.graph_rtee = rtree.index.Index()
        for i, v in enumerate(self.subdivide_points):
            x, y = v
            # hack to insert single points
            self.graph_rtee.insert(i, (x, y, x, y))
        # kdtree for spherical query
        self.graph_kdtree = scipy.spatial.KDTree(self.subdivide_points)

        # pre-exclude points near crossover points
        crossover_exclude_radius = 4
        exclude_indices = set()
        for p in self.crossover_points:
            nearby_indices = self.graph_kdtree.query_ball_point(p, crossover_exclude_radius)
            exclude_indices.update(nearby_indices)
        self.exclude_indices = exclude_indices

        # Find intersection points, these will always be kept in nms
        itsc_indices = set()
        point_num = len(self.full_graph_subdivide.vs)
        for i in range(point_num):
            if self.full_graph_subdivide.degree(i) != 2:
                itsc_indices.add(i)
        self.nms_score_override = np.zeros((point_num, ), dtype=np.float32)
        self.nms_score_override[np.array(list(itsc_indices))] = 2.0  # itsc points will always be kept

        # Points near crossover and intersections are interesting.
        # they will be more frequently sampled
        interesting_indices = set()
        interesting_radius = 32
        # near itsc
        for i in itsc_indices:
            p = self.subdivide_points[i]
            nearby_indices = self.graph_kdtree.query_ball_point(p, interesting_radius)
            interesting_indices.update(nearby_indices)
        for p in self.crossover_points:
            nearby_indices = self.graph_kdtree.query_ball_point(np.array(p), interesting_radius)
            interesting_indices.update(nearby_indices)
        self.sample_weights = np.full((point_num, ), 0.1, dtype=np.float32)
        self.sample_weights[list(interesting_indices)] = 0.9
    
    def sample_patch(self, patch, rot_index = 0):
        (x0, y0), (x1, y1) = patch
        query_box = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        patch_indices_all = set(self.graph_rtee.intersection(query_box))
        patch_indices = patch_indices_all - self.exclude_indices

        # Use NMS to downsample, params shall resemble inference time
        patch_indices = np.array(list(patch_indices))
        if len(patch_indices) == 0:
            # print("==== Patch is empty ====")
            # this shall be rare, but if no points in side the patch, return null stuff
            sample_num = self.config.TOPO_SAMPLE_NUM
            max_nbr_queries = self.config.MAX_NEIGHBOR_QUERIES
            fake_points = np.array([[0.0, 0.0]], dtype=np.float32)
            fake_sample = ([[0, 0]] * max_nbr_queries, [False] * max_nbr_queries, [False] * max_nbr_queries)
            return fake_points, [fake_sample] * sample_num

        patch_points = self.subdivide_points[patch_indices, :]
        
        # random scores to emulate different random configurations that all share a
        # similar spacing between sampled points
        # raise scores for intersction points so they are always kept
        nms_scores = np.random.uniform(low=0.9, high=1.0, size=patch_indices.shape[0])
        nms_score_override = self.nms_score_override[patch_indices]
        nms_scores = np.maximum(nms_scores, nms_score_override)
        nms_radius = self.config.ROAD_NMS_RADIUS
        
        # kept_indces are into the patch_points array
        nmsed_points, kept_indices = graph_utils.nms_points(patch_points, nms_scores, radius=nms_radius, return_indices=True)
        # now this is into the subdivide graph
        nmsed_indices = patch_indices[kept_indices]
        nmsed_point_num = nmsed_points.shape[0]


        sample_num = self.config.TOPO_SAMPLE_NUM  # has to be greater than 1
        sample_weights = self.sample_weights[nmsed_indices]
        # indices into the nmsed points in the patch
        sample_indices_in_nmsed = np.random.choice(
            np.arange(start=0, stop=nmsed_points.shape[0], dtype=np.int32),
            size=sample_num, replace=True, p=sample_weights / np.sum(sample_weights))
        # indices into the subdivided graph
        sample_indices = nmsed_indices[sample_indices_in_nmsed]
        
        radius = self.config.NEIGHBOR_RADIUS
        max_nbr_queries = self.config.MAX_NEIGHBOR_QUERIES  # has to be greater than 1
        nmsed_kdtree = scipy.spatial.KDTree(nmsed_points)
        sampled_points = self.subdivide_points[sample_indices, :]
        # [n_sample, n_nbr]
        # k+1 because the nearest one is always self
        knn_d, knn_idx = nmsed_kdtree.query(sampled_points, k=max_nbr_queries + 1, distance_upper_bound=radius)


        samples = []

        for i in range(sample_num):
            source_node = sample_indices[i]
            valid_nbr_indices = knn_idx[i, knn_idx[i, :] < nmsed_point_num]
            valid_nbr_indices = valid_nbr_indices[1:] # the nearest one is self so remove
            target_nodes = [nmsed_indices[ni] for ni in valid_nbr_indices]  

            ### BFS to find immediate neighbors on graph
            reached_nodes = graph_utils.bfs_with_conditions(self.full_graph_subdivide, source_node, set(target_nodes), radius // self.subdivide_resolution)
            shall_connect = [t in reached_nodes for t in target_nodes]
            ###

            pairs = []
            valid = []
            source_nmsed_idx = sample_indices_in_nmsed[i]
            for target_nmsed_idx in valid_nbr_indices:
                pairs.append((source_nmsed_idx, target_nmsed_idx))
                valid.append(True)

            # zero-pad
            for i in range(len(pairs), max_nbr_queries):
                pairs.append((source_nmsed_idx, source_nmsed_idx))
                shall_connect.append(False)
                valid.append(False)

            samples.append((pairs, shall_connect, valid))

        # Transform points
        # [N, 2]
        nmsed_points -= np.array([x0, y0])[np.newaxis, :]
        # homo for rot
        # [N, 3]
        nmsed_points = np.concatenate([nmsed_points, np.ones((nmsed_point_num, 1), dtype=nmsed_points.dtype)], axis=1)
        trans = np.array([
            [1, 0, -0.5 * self.config.PATCH_SIZE],
            [0, 1, -0.5 * self.config.PATCH_SIZE],
            [0, 0, 1],
        ], dtype=np.float32)
        # ccw 90 deg in img (x, y)
        rot = np.array([
            [0, 1, 0],
            [-1, 0, 0],
            [0, 0, 1],
        ], dtype=np.float32)
        nmsed_points = nmsed_points @ trans.T @ np.linalg.matrix_power(rot.T, rot_index) @ np.linalg.inv(trans.T)
        nmsed_points = nmsed_points[:, :2]
            
        # Add noise
        noise_scale = 1.0  # pixels
        nmsed_points += np.random.normal(0.0, noise_scale, size=nmsed_points.shape)

        return nmsed_points, samples
    

def test_graph_label_generator():
    if not os.path.exists('debug'):
        os.mkdir('debug')

        # coord_transform = lambda v : v[:, ::-1]
    rgb = read_rgb_img(rgb_path)
    config = addict.Dict()
    config.PATCH_SIZE = 256
    config.ROAD_NMS_RADIUS = 16
    config.TOPO_SAMPLE_NUM = 4
    config.NEIGHBOR_RADIUS = 64
    config.MAX_NEIGHBOR_QUERIES = 16
    gen = GraphLabelGenerator(config, gt_graph, coord_transform)
    patch = ((x0, y0), (x1, y1)) = ((64, 64), (64+config.PATCH_SIZE, 64+config.PATCH_SIZE))
    test_num = 64
    for i in range(test_num):
        rot_index = np.random.randint(0, 4)
        points, samples = gen.sample_patch(patch, rot_index=rot_index)
        rgb_patch = rgb[y0:y1, x0:x1, ::-1].copy()
        rgb_patch = np.rot90(rgb_patch, rot_index, (0, 1)).copy()
        for pairs, shall_connect, valid in samples:
            color = tuple(int(c) for c in np.random.randint(0, 256, size=3))

            for (src, tgt), connected, is_valid in zip(pairs, shall_connect, valid):
                if not is_valid:
                    continue
                p0, p1 = points[src], points[tgt]
                cv2.circle(rgb_patch, p0.astype(np.int32), 4, color, -1)
                cv2.circle(rgb_patch, p1.astype(np.int32), 2, color, -1)
                if connected:
                    cv2.line(
                        rgb_patch,
                        (int(p0[0]), int(p0[1])),
                        (int(p1[0]), int(p1[1])),
                        (255, 255, 255),
                        1,
                    )
        cv2.imwrite(f'debug/viz_{i}.png', rgb_patch)

        
def graph_collate_fn(batch):
    keys = batch[0].keys()
    collated = {}
    for key in keys:
        if key == 'graph_points':
            tensors = [item[key] for item in batch]
            max_point_num = max([x.shape[0] for x in tensors])
            padded = []
            for x in tensors:
                pad_num = max_point_num - x.shape[0]
                padded_x = torch.concat([x, torch.zeros(pad_num, 2)], dim=0)
                padded.append(padded_x)
            collated[key] = torch.stack(padded, dim=0)
        elif key == 'tile_class':
            # Handle string values for tile_class separately
            collated[key] = [item[key] for item in batch]
        else:
            collated[key] = torch.stack([item[key] for item in batch], dim=0)
    return collated



class SatMapDataset(Dataset):
    def __init__(self, config, is_train, dev_run=False):
        self.config = config
        
        assert self.config.DATASET in {'os'}
        
        
        self.IMAGE_SIZE = 256
        self.SAMPLE_MARGIN = 0
        
        # Get dataset_id from config or use default
        dataset_id = getattr(self.config, 'DATASET_ID', 'default')
        dataset_dir = f'./os/{dataset_id}'
        
        # Initialize DatasetHandler with the datasset directory and enable GCP download
        dataset_handler = DatasetHandler(dataset_dir, download_from_gcs=False)
        self.dataset_handler = dataset_handler  # Store reference to dataset_handler
        data_config = load_data_config(dataset_dir, self.config)
        self.config.DATA_CONFIG = data_config
        
        # Check if this is a composite dataset
        self.is_composite = False
        if hasattr(data_config, 'metadata') and data_config.metadata.get('is_composite', False):
            self.is_composite = True
            # Initialize dict to store tile classes
            self.tile_classes = {}
            # Load dataset index to get tile metadata
            self.dataset_index = dataset_handler.dataset_index
            logging.info(f"Working with a composite dataset: {dataset_id}")
        
        # Ensure dataset exists locally, DatasetHandler will download from GCP if needed
        if not os.path.exists(dataset_dir) or len(dataset_handler.get_all_tile_ids()) == 0:
            print(f"Dataset {dataset_id} not found locally, attempting to download from GCP...")
            # DatasetHandler automatically attempts download in its initialization
            if not os.path.exists(dataset_dir) or len(dataset_handler.get_all_tile_ids()) == 0:
                raise ValueError(f"Failed to download dataset {dataset_id} from GCP")
            print(f"Successfully downloaded dataset {dataset_id} from GCP")
        
        # Set patterns for file paths based on the dataset structure from DatasetHandler
        rgb_pattern = f'{dataset_dir}/{{}}/raster.png'
        keypoint_mask_pattern = f'{dataset_dir}/{{}}/keypoints.png'
        road_mask_pattern = f'{dataset_dir}/{{}}/road_mask.png'
        gt_graph_pattern = f'{dataset_dir}/{{}}/graph.json'
        metadata_pattern = f'{dataset_dir}/{{}}/metadata.json'  # Pattern for metadata files
        coord_transform = None
        
        # Get data split or generate one if it doesn't exist
        data_split = dataset_handler.get_data_split()
        if data_split is None:
            print(f"Creating new data split for dataset {dataset_id}")
            data_split = dataset_handler.generate_data_split()
            
        train, val, test = data_split['train'], data_split['validation'], data_split['test']
        
        self.is_train = is_train

        train_split = train + val
        test_split = test

        tile_indices = train_split if self.is_train else test_split
        self.tile_indices = tile_indices
        
        # Stores all imgs in memory.
        self.rgbs, self.keypoint_masks, self.road_masks = [], [], []
        # For graph label generation.
        self.graph_label_generators = []

        ##### FAST DEBUG
        if dev_run:
            tile_indices = tile_indices[:4]
        ##### FAST DEBUG

        # Keep track of which index corresponds to which tile
        self.idx_to_tile = {}
        processed_tile_count = 0
        # Dictionary to store metadata for each tile
        self.tile_metadata = {}

        for tile_idx in tile_indices:
            # print(f'loading tile {tile_idx}')
            rgb_path = rgb_pattern.format(tile_idx)
            road_mask_path = road_mask_pattern.format(tile_idx)
            keypoint_mask_path = keypoint_mask_pattern.format(tile_idx)
            metadata_path = metadata_pattern.format(tile_idx)
            
    
            try:
                with open(gt_graph_pattern.format(tile_idx),'r') as jf:
                    gt_graph_adj = json.load(jf)
            except FileNotFoundError:
                print(f'===== skipped missing tile {tile_idx} =====')
                continue
            except json.JSONDecodeError:
                print(f'===== skipped corrupt graph file for tile {tile_idx} =====')
                continue
                    
            try:
                self.rgbs.append(read_rgb_img(rgb_path))
                self.road_masks.append(cv2.imread(road_mask_path, cv2.IMREAD_GRAYSCALE))
                self.keypoint_masks.append(cv2.imread(keypoint_mask_path, cv2.IMREAD_GRAYSCALE))
                graph_label_generator = GraphLabelGenerator(config, gt_graph_adj, coord_transform)
                self.graph_label_generators.append(graph_label_generator)
                
                # Load metadata if it exists
                try:
                    with open(metadata_path, 'r') as mf:
                        self.tile_metadata[processed_tile_count] = json.load(mf)
                except (FileNotFoundError, json.JSONDecodeError):
                    print(f'===== warning: no valid metadata for tile {tile_idx} =====')
                    self.tile_metadata[processed_tile_count] = {}
                
                # Store mapping from processed index to tile_idx
                self.idx_to_tile[processed_tile_count] = tile_idx
                processed_tile_count += 1
                
                # Store tile class for composite datasets
                if self.is_composite and tile_idx in self.dataset_index['tiles']:
                    tile_metadata = self.dataset_index['tiles'][tile_idx]
                    if 'source_dataset' in tile_metadata:
                        self.tile_classes[tile_idx] = tile_metadata['source_dataset']
                
            except Exception as e:
                print(f'===== error loading tile {tile_idx}: {str(e)} =====')
                continue
                
        if self.is_composite:
            self.class_distribution = self.get_class_distribution()
            logging.info(f'Class distribution in {dataset_id}: {self.class_distribution}')
            
            # Calculate class weights for better logging
            total_samples = sum(self.class_distribution.values())
            self.class_weights = {cls: total_samples / count for cls, count in self.class_distribution.items()}
            logging.info(f'Class weights: {self.class_weights}')
            
        # Update tile_indices to only include successfully loaded tiles
        self.tile_indices = [self.idx_to_tile[i] for i in range(processed_tile_count)]
        
        self.sample_min = self.SAMPLE_MARGIN
        self.sample_max = self.IMAGE_SIZE - (self.config.PATCH_SIZE + self.SAMPLE_MARGIN)

        if not self.is_train:
            eval_patches_per_edge = math.ceil((self.IMAGE_SIZE - 2 * self.SAMPLE_MARGIN) / self.config.PATCH_SIZE)
            self.eval_patches = []
            for i in range(len(self.rgbs)):
                self.eval_patches += get_patch_info_one_img(
                    i, self.IMAGE_SIZE, self.SAMPLE_MARGIN, self.config.PATCH_SIZE, eval_patches_per_edge
                )
        
    def get_tile_class(self, tile_idx):
        """Return the class/source dataset of the given tile.
        
        Args:
            tile_idx: The tile ID to query
            
        Returns:
            The source_dataset (class) of the tile if available, None otherwise
        """
        if not self.is_composite:
            return None
        
        return self.tile_classes.get(tile_idx, None)
    
    def get_class_distribution(self):
        """Count the total number of each class in the dataset.
        
        Returns:
            A dictionary with class names as keys and counts as values
        """
        if not self.is_composite:
            return {}
        
        class_counts = {}
        for tile_idx, class_name in self.tile_classes.items():
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
        return class_counts
    
    def calculate_sample_weights(self):
        """Calculate sample weights based on class distribution.
        
        Returns:
            A list of weights, one per sample
        """
        if not self.is_composite:
            return [1.0] * len(self)
            
        # Calculate class weights (inversely proportional to frequency)
        total_samples = sum(self.class_distribution.values())
        class_weights = {cls: total_samples / count for cls, count in self.class_distribution.items()}
        
        # Create sample weights array
        sample_weights = []
        for i in range(len(self)):
            tile_idx = self.tile_indices[i % len(self.tile_indices)]  # Handle case when __len__ returns a different value
            tile_class = self.get_tile_class(tile_idx)
            if tile_class and tile_class in class_weights:
                sample_weights.append(class_weights[tile_class])
            else:
                sample_weights.append(1.0)
                
        return sample_weights

    def __len__(self):
        if self.is_train:
            # Return the actual number of images in the dataset
            return len(self.rgbs)
        else:
            return len(self.eval_patches)

    def __getitem__(self, idx):
        # Sample a patch using the provided index
        if self.is_train:
            # Use the provided index to select the image
            img_idx = idx
            # Get corresponding tile_idx
            tile_idx = self.idx_to_tile.get(img_idx)
            # Still randomly select the patch position within the image
            begin_x = np.random.randint(low=self.sample_min, high=self.sample_max+1)
            begin_y = np.random.randint(low=self.sample_min, high=self.sample_max+1)
            end_x, end_y = begin_x + self.config.PATCH_SIZE, begin_y + self.config.PATCH_SIZE
        else:
            # Returns eval patch
            img_idx, (begin_x, begin_y), (end_x, end_y) = self.eval_patches[idx]
            tile_idx = self.idx_to_tile.get(img_idx)
            
        # Get tile class if this is a composite dataset
        tile_class = None
        if self.is_composite and tile_idx is not None:
            tile_class = self.get_tile_class(tile_idx)
        
        # Get metadata for this tile
        metadata = self.tile_metadata.get(img_idx, {})
        
        # Crop patch imgs and masks
        rgb_patch = self.rgbs[img_idx][begin_y:end_y, begin_x:end_x, :]
        keypoint_mask_patch = self.keypoint_masks[img_idx][begin_y:end_y, begin_x:end_x]
        road_mask_patch = self.road_masks[img_idx][begin_y:end_y, begin_x:end_x]

        # Augmentation
        rot_index = 0
        if self.is_train:
            rot_index = np.random.randint(0, 4)
            # CCW
            rgb_patch = np.rot90(rgb_patch, rot_index, [0,1]).copy()
            keypoint_mask_patch = np.rot90(keypoint_mask_patch, rot_index, [0, 1]).copy()
            road_mask_patch = np.rot90(road_mask_patch, rot_index, [0, 1]).copy()
        
        # Sample graph labels from patch
        patch = ((begin_x, begin_y), (end_x, end_y))
        # points are img (x, y) inside the patch.
        graph_points, topo_samples = self.graph_label_generators[img_idx].sample_patch(patch, rot_index)
        
        pairs, connected, valid = zip(*topo_samples)
        
        
        # rgb: [H, W, 3] 0-255
        # masks: [H, W] 0-1
        return {
            'rgb': torch.tensor(rgb_patch, dtype=torch.float32),
            'keypoint_mask': torch.round(torch.tensor(keypoint_mask_patch, dtype=torch.float32) / 255.0),
            'road_mask': torch.round(torch.tensor(road_mask_patch, dtype=torch.float32) / 255.0),
            
            'graph_points': torch.tensor(graph_points, dtype=torch.float32),
            'pairs': torch.tensor(pairs, dtype=torch.int32),
            'connected': torch.tensor(connected, dtype=torch.bool),
            'valid': torch.tensor(valid, dtype=torch.bool),
            'tile_class': tile_class,
            'metadata': metadata,
        }


if __name__ == '__main__':
    test_graph_label_generator()
    # train, val, test = cityscale_data_partition()
    # print(f'cityscale train {len(train)} val {len(val)} test {len(test)}')
    # train, val, test = spacenet_data_partition()
    # print(f'spacenet train {len(train)} val {len(val)} test {len(test)}')
