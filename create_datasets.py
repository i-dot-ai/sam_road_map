import asyncio
import json
import logging
import math
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import aiohttp
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
import requests
from dotenv import load_dotenv
import networkx as nx

from google.cloud import storage
from PIL import Image
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv()


matplotlib.use("Agg")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple token bucket rate limiter."""

    def __init__(self, rate: float, burst: int = 1):
        """
        Initialize the rate limiter.

        Args:
            rate: Maximum requests per second
            burst: Maximum burst size (token bucket capacity)
        """
        self.rate = rate  # tokens per second
        self.burst = burst  # maximum token bucket size
        self.tokens = burst  # current token count
        self.updated_at = time.monotonic()
        self.lock = asyncio.Lock()

    async def acquire(self):
        """Acquire a token, waiting if necessary."""
        async with self.lock:
            while True:
                # Add new tokens based on elapsed time
                now = time.monotonic()
                elapsed = now - self.updated_at
                new_tokens = elapsed * self.rate
                self.tokens = min(self.burst, self.tokens + new_tokens)
                self.updated_at = now

                if self.tokens >= 1:
                    # Token available
                    self.tokens -= 1
                    return

                # Calculate wait time until next token
                wait_time = (1 - self.tokens) / self.rate
                await asyncio.sleep(wait_time)


class RetryWithBackoff:
    """Utility class for retrying operations with exponential backoff."""

    def __init__(
        self,
        max_retries: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        jitter: bool = True,
        retry_on_exceptions: Tuple = (Exception,),
    ):
        """
        Initialize the retry handler.

        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Base delay in seconds
            max_delay: Maximum delay in seconds
            jitter: Whether to add random jitter to delay
            retry_on_exceptions: Exceptions to retry on
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter
        self.retry_on_exceptions = retry_on_exceptions

    async def execute(self, coro_func, *args, **kwargs):
        """Execute a coroutine with retry logic."""
        retries = 0
        last_exception = None

        while retries <= self.max_retries:
            try:
                return await coro_func(*args, **kwargs)
            except self.retry_on_exceptions as e:
                retries += 1
                last_exception = e

                if retries > self.max_retries:
                    break

                # Calculate exponential backoff delay
                delay = min(self.max_delay, self.base_delay * (2 ** (retries - 1)))

                # Add jitter if enabled (prevents thundering herd)
                if self.jitter:
                    delay = delay * (0.5 + random.random())

                logger.debug(
                    f"Retry {retries}/{self.max_retries} after error: {str(e)}. Waiting {delay:.2f}s"
                )
                await asyncio.sleep(delay)

        # If we got here, all retries failed
        raise last_exception


class DatasetHandler:
    """
    Class for managing tile datasets. Handles saving and retrieving tile data
    including raster images, road masks, keypoints, and graph data.
    """

    def __init__(
        self,
        dataset_dir: Union[str, Path],
        download_from_gcs: bool = False,
        dataset_id: Optional[str] = None,
    ):
        """
        Initialize the dataset handler with the dataset directory.

        Args:
            dataset_dir: Directory where dataset is/will be stored
            download_from_gcs: Whether to download the dataset from Google Cloud Storage if not found locally
            dataset_id: Optional dataset ID to use for GCS path determination. If None, will use basename of dataset_dir
        """
        self.dataset_dir = Path(dataset_dir)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.dataset_dir / "index.json"
        self.download_from_gcs = download_from_gcs
        self.dataset_id = dataset_id or os.path.basename(str(self.dataset_dir))
        self.bucket_name = "extract-general"
        self.dataset_index = self._load_or_create_index()
        self.upload_to_gcs_enabled = (
            False  # Default is False, can be set after initialization
        )
        self.gcs_client = None  # Will be initialized when needed
        self.gcs_bucket = None  # Will be initialized when needed
        self.gcs_base_path = None  # Will be initialized when needed

    def _load_or_create_index(self) -> Dict[str, Any]:
        """
        Load existing index or create a new one if it doesn't exist.
        """

        if self.download_from_gcs:
            # Try to download the dataset from GCS if enabled
            if self._download_from_gcs():
                # After download, try loading index again
                if self.index_file.exists():
                    with open(self.index_file, "r") as f:
                        return json.load(f)

        elif self.index_file.exists():
            with open(self.index_file, "r") as f:
                return json.load(f)

        # Create new index if it doesn't exist or couldn't be downloaded
        index = {
            "tiles": {},
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "tile_count": 0,
                "is_composite": False,  # Flag to indicate if this is a composite dataset
            },
        }
        self._save_index(index)
        return index

    def _save_index(self, index: Optional[Dict[str, Any]] = None) -> None:
        """
        Save the index to disk.
        """
        if index is None:
            index = self.dataset_index

        with open(self.index_file, "w") as f:
            json.dump(index, f, indent=2)

    def _download_from_gcs(
        self, bucket_name: Optional[str] = None, source_prefix: Optional[str] = None
    ) -> bool:
        """
        Download the dataset from a Google Cloud Storage bucket.

        Args:
            bucket_name: Name of the GCS bucket. If None, tries to get from metadata
            source_prefix: Path prefix in the bucket. If None, uses the standard path based on dataset_id

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check if we have GCS information in metadata
            if bucket_name is None or source_prefix is None:
                # Try to get dataset info from environment variables
                bucket_name = self.bucket_name
                source_prefix = source_prefix or f"sam_road/datasets/{self.dataset_id}"

            logger.info(f"Downloading dataset from gs://{bucket_name}/{source_prefix}")

            # Initialize GCS client
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)

            # List all objects with the prefix
            blobs = list(bucket.list_blobs(prefix=source_prefix))
            total_files = len(blobs)

            if total_files == 0:
                logger.warning(f"No files found at gs://{bucket_name}/{source_prefix}")
                return False

            logger.info(f"Found {total_files} files to download")

            # Create local directories and download files
            file_count = 0
            # Create progress bar
            with tqdm(total=total_files, desc="Downloading files", unit="file") as pbar:
                for blob in blobs:
                    # Skip directory markers
                    if blob.name.endswith("/"):
                        pbar.update(1)
                        continue

                    # Get relative path
                    rel_path = blob.name[len(source_prefix) :].lstrip("/")
                    if not rel_path:  # Skip the prefix directory itself
                        pbar.update(1)
                        continue

                    # Determine local path
                    local_path = self.dataset_dir / rel_path

                    # Create parent directories if needed
                    local_path.parent.mkdir(parents=True, exist_ok=True)

                    # Download the file
                    blob.download_to_filename(str(local_path))
                    file_count += 1
                    pbar.update(1)

            logger.info(f"Successfully downloaded {file_count} files from GCS")
            return file_count > 0

        except Exception as e:
            logger.error(f"Error downloading dataset from GCS: {str(e)}")
            return False

    def enable_gcs_upload(self, bucket_name: str, base_path: Optional[str] = None):
        """
        Enable automatic uploads to GCS for all saved files.

        Args:
            bucket_name: Name of the GCS bucket
            base_path: Base path within the bucket (if None, uses standard path based on dataset_id)
        """
        self.upload_to_gcs_enabled = True
        self.gcs_bucket_name = bucket_name
        self.gcs_base_path = base_path or f"sam_road/datasets/{self.dataset_id}"

        # Initialize GCS client and bucket
        try:
            if self.gcs_client is None:
                self.gcs_client = storage.Client()
            self.gcs_bucket = self.gcs_client.bucket(bucket_name)
            logger.info(
                f"GCS uploads enabled to gs://{bucket_name}/{self.gcs_base_path}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize GCS client: {e}")
            self.upload_to_gcs_enabled = False
            raise

    def upload_file_to_gcs(
        self, local_path: Union[str, Path], max_retries: int = 10
    ) -> Optional[str]:
        """
        Upload a single file to GCS with retry logic.

        Args:
            local_path: Path to the local file
            max_retries: Maximum number of retry attempts

        Returns:
            GCS URL of the uploaded file, or None if upload failed
        """
        if not self.upload_to_gcs_enabled or self.gcs_bucket is None:
            return None

        local_path = Path(local_path)
        # Get relative path from dataset directory
        rel_path = local_path.relative_to(self.dataset_dir)
        # Create GCS path
        gcs_path = f"{self.gcs_base_path}/{rel_path}"

        # Create retry parameters
        retry_count = 0
        base_delay = 1.0
        max_delay = 60.0

        while retry_count <= max_retries:
            try:
                # Upload file
                blob = self.gcs_bucket.blob(gcs_path)
                blob.upload_from_filename(str(local_path))

                # Return GCS URL
                gcs_url = f"gs://{self.gcs_bucket_name}/{gcs_path}"
                logger.debug(f"Uploaded {local_path} to {gcs_url}")
                return gcs_url
            except Exception as e:
                retry_count += 1

                if retry_count > max_retries:
                    logger.error(
                        f"Error uploading {local_path} to GCS after {max_retries} retries: {e}"
                    )
                    return None

                # Calculate exponential backoff delay
                delay = min(max_delay, base_delay * (2 ** (retry_count - 1)))
                # Add jitter
                delay = delay * (0.5 + random.random())

                logger.warning(
                    f"GCS upload failed for {local_path}, retry {retry_count}/{max_retries} after {delay:.2f}s. Error: {e}"
                )
                time.sleep(delay)

    def verify_and_sync_gcs_uploads(self) -> Dict[str, int]:
        """
        Verify that all files in the dataset have been uploaded to GCS.
        If any files are missing, upload them with retry logic.

        Returns:
            Dict with statistics about the verification process
        """
        if not self.upload_to_gcs_enabled or self.gcs_bucket is None:
            logger.warning("GCS uploads not enabled, skipping verification")
            return {"status": "skipped", "reason": "uploads_not_enabled"}

        logger.info("Starting verification of GCS uploads...")

        stats = {
            "total_files": 0,
            "already_uploaded": 0,
            "newly_uploaded": 0,
            "failed_uploads": 0,
        }

        try:
            # Get list of all files in GCS under this dataset path
            gcs_files = set()
            blobs = self.gcs_bucket.list_blobs(prefix=self.gcs_base_path)
            for blob in blobs:
                gcs_files.add(blob.name)

            logger.info(f"Found {len(gcs_files)} files already in GCS")

            # Walk through all local files
            all_local_files = []
            for root, dirs, files in os.walk(self.dataset_dir):
                for file in files:
                    all_local_files.append(os.path.join(root, file))

            # Create progress bar for verification process
            with tqdm(
                total=len(all_local_files), desc="Verifying GCS uploads", unit="file"
            ) as pbar:
                for local_path_str in all_local_files:
                    local_path = Path(local_path_str)
                    stats["total_files"] += 1

                    # Get relative path from dataset directory
                    try:
                        rel_path = local_path.relative_to(self.dataset_dir)
                        # Create GCS path
                        gcs_path = f"{self.gcs_base_path}/{rel_path}"

                        # Check if file exists in GCS
                        if gcs_path in gcs_files:
                            stats["already_uploaded"] += 1
                        else:
                            # Upload file with retry
                            logger.info(f"File missing in GCS, uploading: {local_path}")
                            result = self.upload_file_to_gcs(local_path, max_retries=5)
                            if result:
                                stats["newly_uploaded"] += 1
                                logger.info(
                                    f"Successfully uploaded missing file: {local_path}"
                                )
                            else:
                                stats["failed_uploads"] += 1
                                logger.error(
                                    f"Failed to upload missing file: {local_path}"
                                )
                    except Exception as e:
                        logger.error(f"Error processing file {local_path}: {e}")
                        stats["failed_uploads"] += 1

                    pbar.update(1)

            logger.info(f"GCS verification complete - Stats: {stats}")
            return stats

        except Exception as e:
            logger.error(f"Error during GCS verification: {e}")
            stats["error"] = str(e)
            return stats

    def save_tile_data(
        self,
        tile_id: str,
        raster_img: Optional[np.ndarray] = None,
        road_mask_img: Optional[np.ndarray] = None,
        keypoints_img: Optional[np.ndarray] = None,
        graph_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        raster_bytes: Optional[bytes] = None,
    ) -> Dict[str, str]:
        """
        Save all data for a single tile.

        Args:
            tile_id: Unique identifier for the tile
            raster_img: Raster image as numpy array
            road_mask_img: Road mask image as numpy array
            keypoints_img: Keypoints visualization as numpy array
            graph_data: Graph structure data
            metadata: Additional metadata for the tile
            raster_bytes: Raw raster image bytes (alternative to raster_img)

        Returns:
            Dict with paths to all saved files
        """
        # Create directory for this tile
        tile_dir = self.dataset_dir / tile_id
        tile_dir.mkdir(exist_ok=True)

        file_paths = {}
        gcs_paths = {}

        # Handle raster image
        if raster_img is not None or raster_bytes is not None:
            raster_path = tile_dir / "raster.png"

            if raster_img is not None:
                Image.fromarray(raster_img).save(raster_path)
            elif raster_bytes is not None:
                # Create image from bytes
                img = Image.open(Image.io.BytesIO(raster_bytes))
                img.save(raster_path, format="PNG", compress_level=0)

            file_paths["raster"] = str(raster_path)

            # Upload to GCS if enabled
            if self.upload_to_gcs_enabled:
                gcs_url = self.upload_file_to_gcs(raster_path)
                if gcs_url:
                    gcs_paths["raster"] = gcs_url

        # Handle road mask
        if road_mask_img is not None:
            road_mask_path = tile_dir / "road_mask.png"
            if len(road_mask_img.shape) == 2 or road_mask_img.shape[2] == 1:
                # Single channel
                Image.fromarray(road_mask_img).save(road_mask_path)
            else:
                # RGB or RGBA
                Image.fromarray(road_mask_img).save(road_mask_path)
            file_paths["road_mask"] = str(road_mask_path)

            # Upload to GCS if enabled
            if self.upload_to_gcs_enabled:
                gcs_url = self.upload_file_to_gcs(road_mask_path)
                if gcs_url:
                    gcs_paths["road_mask"] = gcs_url

        # Handle keypoints image
        if keypoints_img is not None:
            keypoints_path = tile_dir / "keypoints.png"
            Image.fromarray(keypoints_img).save(keypoints_path)
            file_paths["keypoints"] = str(keypoints_path)

            # Upload to GCS if enabled
            if self.upload_to_gcs_enabled:
                gcs_url = self.upload_file_to_gcs(keypoints_path)
                if gcs_url:
                    gcs_paths["keypoints"] = gcs_url

        # Handle graph data
        if graph_data is not None:
            graph_path = tile_dir / "graph.json"
            with open(graph_path, "w") as f:
                json.dump(graph_data, f, indent=2)
            file_paths["graph"] = str(graph_path)

            # Upload to GCS if enabled
            if self.upload_to_gcs_enabled:
                gcs_url = self.upload_file_to_gcs(graph_path)
                if gcs_url:
                    gcs_paths["graph"] = gcs_url

        # Handle metadata
        if metadata is not None:
            metadata_path = tile_dir / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            file_paths["metadata"] = str(metadata_path)

            # Upload to GCS if enabled
            if self.upload_to_gcs_enabled:
                gcs_url = self.upload_file_to_gcs(metadata_path)
                if gcs_url:
                    gcs_paths["metadata"] = gcs_url

        # Update index
        tile_entry = {
            "files": file_paths,
            "metadata": metadata,
            "node_count": len(graph_data["nodes"]) if graph_data else 0,
            "edge_count": len(graph_data["edges"]) if graph_data else 0,
        }

        # Add GCS paths if any were uploaded
        if gcs_paths:
            tile_entry["gcs_files"] = gcs_paths

        self.dataset_index["tiles"][tile_id] = tile_entry
        self.dataset_index["metadata"]["tile_count"] = len(self.dataset_index["tiles"])

        # Upload the updated index to GCS too if enabled
        self._save_index()
        if self.upload_to_gcs_enabled:
            self.upload_file_to_gcs(self.index_file)

        return file_paths

    def get_tile_data(self, tile_id: str) -> Dict[str, Any]:
        """
        Retrieve all data for a specific tile.

        Args:
            tile_id: Unique identifier for the tile

        Returns:
            Dict containing all data for the tile
        """
        if tile_id not in self.dataset_index["tiles"]:
            raise ValueError(f"Tile ID {tile_id} not found in dataset")

        tile_info = self.dataset_index["tiles"][tile_id]
        result = {"id": tile_id}

        # Load raster image
        if "raster" in tile_info["files"]:
            raster_path = tile_info["files"]["raster"]
            result["raster_img"] = np.array(Image.open(raster_path))

        # Load road mask
        if "road_mask" in tile_info["files"]:
            road_mask_path = tile_info["files"]["road_mask"]
            result["road_mask_img"] = np.array(Image.open(road_mask_path))

        # Load keypoints image
        if "keypoints" in tile_info["files"]:
            keypoints_path = tile_info["files"]["keypoints"]
            result["keypoints_img"] = np.array(Image.open(keypoints_path))

        # Load graph data
        if "graph" in tile_info["files"]:
            graph_path = tile_info["files"]["graph"]
            with open(graph_path, "r") as f:
                result["graph_data"] = json.load(f)

        # Load metadata
        if "metadata" in tile_info["files"]:
            metadata_path = tile_info["files"]["metadata"]
            with open(metadata_path, "r") as f:
                result["metadata"] = json.load(f)
        elif "metadata" in tile_info:
            result["metadata"] = tile_info["metadata"]

        # Extract points data
        if "graph_data" in result and "nodes" in result["graph_data"]:
            result["points"] = list(result["graph_data"]["nodes"].values())

        return result

    def get_all_tile_ids(self) -> List[str]:
        """
        Get a list of all tile IDs in the dataset.

        Returns:
            List of tile IDs
        """
        return list(self.dataset_index["tiles"].keys())

    def get_dataset_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the dataset.

        Returns:
            Dict with dataset statistics
        """
        stats = {
            "tile_count": len(self.dataset_index["tiles"]),
            "total_nodes": sum(
                tile["node_count"] for tile in self.dataset_index["tiles"].values()
            ),
            "total_edges": sum(
                tile["edge_count"] for tile in self.dataset_index["tiles"].values()
            ),
        }

        if "metadata" in self.dataset_index:
            stats.update(self.dataset_index["metadata"])

        return stats

    def get_batch(self, tile_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Get data for multiple tiles.

        Args:
            tile_ids: List of tile IDs to retrieve

        Returns:
            List of tile data dictionaries
        """
        return [self.get_tile_data(tile_id) for tile_id in tile_ids]

    def upload_to_gcs(
        self, bucket_name: str, destination_prefix: Optional[str] = None
    ) -> bool:
        """
        Upload the entire dataset to a Google Cloud Storage bucket.

        Args:
            bucket_name: Name of the GCS bucket
            destination_prefix: Custom prefix for the destination in the bucket
                               (default: 'sam_road/datasets/{dataset_name}')

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Get dataset name from directory name
            dataset_name = os.path.basename(self.dataset_dir)

            # If no custom prefix is provided, use the default pattern
            if destination_prefix is None:
                destination_prefix = f"sam_road/datasets/{dataset_name}"

            # Use upload_to_gcs_bucket utility function
            success = self._upload_to_gcs_bucket(
                source_directory=str(self.dataset_dir),
                bucket_name=bucket_name,
                destination_prefix=destination_prefix,
            )

            if success:
                logger.info(
                    f"Successfully uploaded dataset to gs://{bucket_name}/{destination_prefix}"
                )
                # Update the dataset index with upload information
                self.dataset_index["metadata"]["gcs_upload"] = {
                    "bucket": bucket_name,
                    "path": destination_prefix,
                    "upload_time": datetime.now().isoformat(),
                }
                self._save_index()
            else:
                logger.error("Failed to upload dataset to GCS bucket")

            return success

        except Exception as e:
            logger.error(f"Error uploading dataset to GCS: {str(e)}")
            return False

    @staticmethod
    def _upload_to_gcs_bucket(
        source_directory: str,
        bucket_name: str,
        destination_prefix: Optional[str] = None,
    ) -> bool:
        """
        Uploads the entire content of a directory to a Google Cloud Storage bucket.

        Args:
            source_directory: Local directory to upload
            bucket_name: Name of the GCS bucket
            destination_prefix: Prefix for the destination in the bucket
                                (essentially a "directory" in the bucket)

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Initialize GCS client
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)

            logger.info(
                f"Starting upload of {source_directory} to GCS bucket {bucket_name}"
            )

            # Walk through all files in the directory
            for root, dirs, files in os.walk(source_directory):
                for file in files:
                    local_path = os.path.join(root, file)

                    # Determine the GCS path (with prefix if provided)
                    rel_path = os.path.relpath(local_path, source_directory)
                    if destination_prefix:
                        gcs_path = f"{destination_prefix}/{rel_path}"
                    else:
                        gcs_path = rel_path

                    # Create a blob and upload the file
                    blob = bucket.blob(gcs_path)
                    blob.upload_from_filename(local_path)
                    logger.debug(
                        f"Uploaded {local_path} to gs://{bucket_name}/{gcs_path}"
                    )

            logger.info(
                f"Successfully uploaded {source_directory} to GCS bucket {bucket_name}"
            )
            return True

        except Exception as e:
            logger.error(f"Error uploading to GCS: {str(e)}")
            return False

    def generate_data_split(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ) -> Dict[str, List[str]]:
        """
        Generate train/val/test split for the dataset in a seeded way.

        Args:
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            test_ratio: Proportion of data for testing
            seed: Random seed for reproducibility

        Returns:
            Dictionary with 'train', 'validation', and 'test' lists of tile IDs
        """
        # Set random seed for reproducibility
        random.seed(seed)

        # Get all tile IDs
        all_tile_ids = self.get_all_tile_ids()

        # Validate that we have tiles to split
        if not all_tile_ids:
            logger.error(
                "ERROR: Cannot create data split - no tiles found in the dataset."
            )
            logger.error(f"Dataset directory: {self.dataset_dir}")
            logger.error(f"Index file exists: {self.index_file.exists()}")
            if self.index_file.exists():
                logger.error(
                    f"Number of tiles in index: {len(self.dataset_index['tiles'])}"
                )
            return {"train": [], "validation": [], "test": []}

        logger.info(f"Creating split for {len(all_tile_ids)} tiles")

        # Validate that ratios sum to 1
        total_ratio = train_ratio + val_ratio + test_ratio
        if not 0.99 <= total_ratio <= 1.01:  # Allow small floating-point error
            raise ValueError(f"Split ratios must sum to 1, got {total_ratio}")

        # Shuffle the tile IDs
        random.shuffle(all_tile_ids)

        # Calculate split sizes
        total_size = len(all_tile_ids)
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)

        # Split the data
        train_ids = all_tile_ids[:train_size]
        val_ids = all_tile_ids[train_size : train_size + val_size]
        test_ids = all_tile_ids[train_size + val_size :]

        # Create split dictionary
        split_dict = {
            "train": sorted(train_ids),
            "validation": sorted(val_ids),
            "test": sorted(test_ids),
        }

        # Debug output
        logger.debug("Split details:")
        logger.debug(f"  Total tiles: {total_size}")
        logger.debug(f"  Train size: {len(train_ids)} ({train_ratio*100:.1f}%)")
        logger.debug(f"  Validation size: {len(val_ids)} ({val_ratio*100:.1f}%)")
        logger.debug(f"  Test size: {len(test_ids)} ({test_ratio*100:.1f}%)")

        if not train_ids:
            logger.warning("WARNING: Train set is empty!")
        if not val_ids:
            logger.warning("WARNING: Validation set is empty!")
        if not test_ids:
            logger.warning("WARNING: Test set is empty!")

        # Print first few examples from each set
        if train_ids:
            logger.debug(f"  Example train IDs: {train_ids[:3]}")
        if val_ids:
            logger.debug(f"  Example validation IDs: {val_ids[:3]}")
        if test_ids:
            logger.debug(f"  Example test IDs: {test_ids[:3]}")

        # Save to JSON in the dataset directory
        split_file = self.dataset_dir / "data_split.json"
        with open(split_file, "w") as f:
            json.dump(split_dict, f, indent=2)

        # Update dataset metadata
        self.dataset_index["metadata"]["data_split"] = {
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "seed": seed,
            "train_count": len(train_ids),
            "validation_count": len(val_ids),
            "test_count": len(test_ids),
        }
        self._save_index()

        logger.info(
            f"Split created with {len(train_ids)} train, {len(val_ids)} validation, and {len(test_ids)} test samples"
        )
        logger.info(f"Split saved to {split_file}")

        return split_dict

    def get_data_split(self) -> Optional[Dict[str, List[str]]]:
        """
        Get the train/val/test split for the dataset.

        Returns:
            Dictionary with 'train', 'validation', and 'test' lists of tile IDs, or None if no split exists
        """
        split_file = self.dataset_dir / "data_split.json"
        if not split_file.exists():
            return None

        with open(split_file, "r") as f:
            return json.load(f)


class MapTilerTileJSON:
    """
    Class for interacting with MapTiler TileJSON API. to get the metadata of tilesets
    """

    def __init__(self):
        # Get API key from environment variables
        self.api_key = os.getenv("MAPTILER_API_KEY")
        if not self.api_key:
            raise ValueError("MAPTILER_API_KEY not found in environment variables")

        self.base_url = "https://api.maptiler.com/tiles"

    def get_tileset_metadata(self, tileset_id="satellite-v2"):
        """
        Fetch the TileJSON metadata for a specific tileset.
        Returns detailed information about the tileset including bounds, zoom levels, and tile URLs.
        """
        url = f"{self.base_url}/{tileset_id}/tiles.json"
        params = {"key": self.api_key}

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()  # Raise exception for bad status codes

            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching TileJSON: {e}")
            return None

    def print_tileset_info(self, tileset_id="satellite-v2"):
        """
        Print formatted information about the tileset.
        """
        metadata = self.get_tileset_metadata(tileset_id)
        if not metadata:
            return

        logger.info("\n=== Tileset Information ===")
        logger.info(f"Name: {metadata.get('name', 'N/A')}")
        logger.info(f"Description: {metadata.get('description', 'N/A')}")
        logger.info(f"Version: {metadata.get('version', 'N/A')}")
        logger.info(f"Attribution: {metadata.get('attribution', 'N/A')}")

        if "bounds" in metadata:
            logger.info(f"Bounds: {metadata['bounds']}")  # [west, south, east, north]

        logger.info(f"Minimum zoom: {metadata.get('minzoom', 'N/A')}")
        logger.info(f"Maximum zoom: {metadata.get('maxzoom', 'N/A')}")

        if "center" in metadata:
            logger.info(
                f"Default center: {metadata['center']}"
            )  # [longitude, latitude, zoom]

        logger.info("\nTile URL template:")
        if "tiles" in metadata and metadata["tiles"]:
            logger.info(metadata["tiles"][0])


def deg2num(lat_deg: float, lon_deg: float, zoom: int) -> Tuple[int, int]:
    """Convert latitude/longitude to tile coordinates"""
    lat_rad = math.radians(lat_deg)
    n = 2.0**zoom
    x = int((lon_deg + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def num2deg(x: float, y: float, zoom: int) -> Tuple[float, float]:
    """Convert tile coordinates to latitude/longitude"""
    n = 2.0**zoom
    lon_deg = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat_deg = math.degrees(lat_rad)
    return lat_deg, lon_deg


def get_tile_bbox(x: int, y: int, zoom: int) -> Dict[str, float]:
    """Get the bounding box for a tile"""
    # Get coordinates for corners
    north, west = num2deg(x, y, zoom)
    south, east = num2deg(x + 1, y + 1, zoom)
    return {"north": north, "south": south, "west": west, "east": east}


def get_tile_scale(lat: float, zoom: int) -> float:
    """Calculate the approximate scale of the tile in meters per pixel
    Based on the fact that MapTiler uses 256x256 pixel tiles"""
    # Earth's circumference at the equator in meters
    earth_circumference = 40075016.686
    # Length of one pixel in meters at this zoom level and latitude
    pixel_scale = (earth_circumference * math.cos(math.radians(lat))) / (
        math.pow(2, zoom + 8)
    )
    return pixel_scale


def create_road_mask(
    bbox: Dict[str, float],
    output_path: Optional[Union[str, Path]] = None,
    keypoints_file: Optional[Union[str, Path]] = None,
    json_file: Optional[Union[str, Path]] = None,
    keep_one_node_in_every: int = 3,
    project_to_osgb: bool = False
) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], Optional[Dict[str, Any]]]:
    """
    Creates a road mask image using OSMnx for the given bounding box.
    Also saves the graph structure and node coordinates in pixel space (256x256).

    Args:
        bbox: Dictionary containing 'north', 'south', 'east', 'west' coordinates in WGS84
        output_path: Path where the road mask should be saved (optional)
        keypoints_file: Path where the keypoints visualization should be saved (optional)
        json_file: Path where the graph data should be saved (optional)
        keep_one_node_in_every: Number of intermediate nodes to skip
        project_to_osgb: Whether to project coordinates to EPSG:27700 (British National Grid)

    Returns:
        tuple: (success, road_mask_img, keypoints_img, graph_data)
            - success: Boolean indicating if the operation was successful
            - road_mask_img: NumPy array of the road mask image (single channel)
            - keypoints_img: NumPy array of the keypoints visualization (RGB)
            - graph_data: Dictionary containing the graph structure
    """
    try:
        # Check if cv2 methods are available
        has_cv2 = all(
            hasattr(cv2, attr)
            for attr in [
                "cvtColor",
                "COLOR_RGB2GRAY",
                "threshold",
                "THRESH_BINARY",
                "imwrite",
            ]
        )

        img_size = 256
        original_backend = plt.get_backend()
        if original_backend != "Agg":
            plt.switch_backend("Agg")

        # Create graph from bbox (always in WGS84)
        left, bottom, right, top = (
            bbox["west"],
            bbox["south"],
            bbox["east"],
            bbox["north"],
        )
        g = ox.graph.graph_from_bbox(
            (left, bottom, right, top), truncate_by_edge=True, network_type="drive"
        )
        
        # Apply coordinate projection if requested
        if project_to_osgb:
            g, proj_bbox, projection_crs = project_graph_and_calculate_bbox(g, 'epsg:27700')
        else:
            # Use original WGS84 bbox
            proj_bbox = bbox
            projection_crs = 'epsg:4326'  # WGS84

        # Create graph data structure with transformed coordinates
        graph_data, nodes_to_add = process_graph_nodes_and_edges(
            g, proj_bbox, projection_crs, img_size, keep_one_node_in_every
        )

        # Add all nodes to NetworkX graph after iteration
        for node_id, node_attrs in nodes_to_add.items():
            g.add_node(node_id, **node_attrs)

        # Save graph data to JSON if path provided
        if json_file:
            with open(json_file, "w") as f:
                json.dump(graph_data, f, indent=2)

        # Generate visualization images
        road_mask_single_channel, keypoints_img = create_visualization_images(
            g, proj_bbox, graph_data, img_size, output_path, keypoints_file, has_cv2
        )

        plt.switch_backend(original_backend)

        return True, road_mask_single_channel, keypoints_img, graph_data

    except Exception as e:
        logger.error(f"Error creating road mask: {e}")
        # If the error is about not finding graph nodes, raise a ValueError
        if "Found no graph nodes within the requested polygon" in str(e):
            raise ValueError("No road data found in the specified area") from e
        return False, None, None, None


def project_graph_and_calculate_bbox(g: nx.MultiDiGraph, to_crs: str) -> Tuple[nx.MultiDiGraph, Dict[str, float], str]:
    """
    Project a graph to a new coordinate system and calculate the bounding box in the new coordinates.
    
    Args:
        g: NetworkX graph to project
        to_crs: Target coordinate reference system (e.g., 'epsg:27700')
        
    Returns:
        Tuple containing:
        - Projected graph
        - Dictionary with projected bounding box (west, south, east, north)
        - CRS string of the projection
    """
    # Project the graph to the target CRS
    g = ox.project_graph(g, to_crs=to_crs)
    
    # Calculate a new bounding box in the projected coordinates
    if len(g.nodes) > 0:
        node = list(g.nodes(data=True))[0]
        min_x, max_x = node[1]['x'], node[1]['x']
        min_y, max_y = node[1]['y'], node[1]['y']
        
        # Find min/max coordinates in the projected graph
        for _, node_data in g.nodes(data=True):
            min_x = min(min_x, node_data['x'])
            max_x = max(max_x, node_data['x'])
            min_y = min(min_y, node_data['y'])
            max_y = max(max_y, node_data['y'])
        
        # Create projected bounding box
        proj_bbox = {
            "west": min_x,
            "south": min_y,
            "east": max_x,
            "north": max_y
        }
    else:
        # If no nodes, create an empty projected bbox
        proj_bbox = {"west": 0, "south": 0, "east": 0, "north": 0}
        logger.warning("No nodes found in graph after projection")
    
    return g, proj_bbox, to_crs


def process_graph_nodes_and_edges(
    g: nx.MultiDiGraph, 
    bbox: Dict[str, float], 
    projection_crs: str,
    img_size: int,
    keep_one_node_in_every: int
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """
    Process a graph's nodes and edges to create a standardized data structure with pixel coordinates.
    
    Args:
        g: NetworkX graph to process
        bbox: Bounding box in the graph's coordinate system
        projection_crs: Coordinate reference system string
        img_size: Size of the output image (square)
        keep_one_node_in_every: Sampling rate for nodes from edge geometries
        
    Returns:
        Tuple containing:
        - Graph data dictionary
        - Dictionary of nodes to add to the original graph
    """
    graph_data: Dict[str, Any] = {"nodes": {}, "edges": []}
    nodes_to_add: Dict[str, Dict[str, Any]] = {}
    
    # Extract values from bbox for readability
    west, south, east, north = bbox["west"], bbox["south"], bbox["east"], bbox["north"]
    
    # First, process all edge geometries to get intermediate points
    node_id_counter = 0

    for u, v, data in g.edges(data=True):
        if "geometry" not in data:
            continue

        coords = list(data["geometry"].coords)
        edge_points: List[str] = []  # Store points for this edge

        for i, (x, y) in enumerate(coords):
            # Skip every other point and points outside the bounding box
            if (
                (
                    i % keep_one_node_in_every != 0
                    and i != 0
                    and i != len(coords) - 1
                )
                or y < south
                or y > north
                or x < west
                or x > east
            ):
                continue

            # Convert to pixel coordinates
            px = ((x - west) / (east - west)) * img_size
            py = (1 - ((y - south) / (north - south))) * img_size

            # Create a unique ID for this point
            point_id = f"edge_point_{node_id_counter}"
            node_id_counter += 1

            # Store node data in both data structures
            node_data = {
                "id": point_id,
                "pixel_coords": [float(px), float(py)],
                "geo_coords": [float(x), float(y)],
                "projection": projection_crs,
            }
            graph_data["nodes"][point_id] = node_data
            nodes_to_add[point_id] = {
                "x": x,
                "y": y,
                "pixel_coords": [float(px), float(py)],
            }
            edge_points.append(point_id)

        # Create edges between consecutive points
        if edge_points:
            # Connect source node to first intermediate point
            if str(u) in graph_data["nodes"]:
                graph_data["edges"].append(
                    {
                        "source": str(u),
                        "target": edge_points[0],
                        "length": float(data.get("length", 0))
                        / (len(edge_points) + 1),
                    }
                )

            # Connect intermediate points
            for i in range(len(edge_points) - 1):
                graph_data["edges"].append(
                    {
                        "source": edge_points[i],
                        "target": edge_points[i + 1],
                        "length": float(data.get("length", 0))
                        / (len(edge_points) + 1),
                    }
                )

            # Connect last intermediate point to target node
            if str(v) in graph_data["nodes"]:
                graph_data["edges"].append(
                    {
                        "source": edge_points[-1],
                        "target": str(v),
                        "length": float(data.get("length", 0))
                        / (len(edge_points) + 1),
                    }
                )

    # Now process junction nodes
    for node in g.nodes():
        # Get coordinates
        x = g.nodes[node]["x"]
        y = g.nodes[node]["y"]

        # Skip nodes outside the bounding box
        if (
            y < south
            or y > north
            or x < west
            or x > east
        ):
            continue

        # Convert to pixel coordinates
        px = ((x - west) / (east - west)) * img_size
        py = (1 - ((y - south) / (north - south))) * img_size

        # Store node data
        node_data = {
            "id": str(node),
            "pixel_coords": [float(px), float(py)],
            "geo_coords": [float(x), float(y)],
            "projection": projection_crs,
        }
        graph_data["nodes"][str(node)] = node_data
        nodes_to_add[str(node)] = {
            "x": x,
            "y": y,
            "pixel_coords": [float(px), float(py)],
        }
        
    return graph_data, nodes_to_add


def create_visualization_images(
    g: nx.MultiDiGraph,
    bbox: Dict[str, float],
    graph_data: Dict[str, Any],
    img_size: int,
    output_path: Optional[Union[str, Path]],
    keypoints_file: Optional[Union[str, Path]],
    has_cv2: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create road mask and keypoints visualization images from a graph.
    
    Args:
        g: NetworkX graph to visualize
        bbox: Bounding box in the graph's coordinate system
        graph_data: Processed graph data dictionary
        img_size: Size of the output image (square)
        output_path: Path to save the road mask image
        keypoints_file: Path to save the keypoints image
        has_cv2: Whether OpenCV is available
        
    Returns:
        Tuple containing:
        - Road mask image as NumPy array
        - Keypoints visualization image as NumPy array
    """
    # Extract values from bbox for readability
    west, south, east, north = bbox["west"], bbox["south"], bbox["east"], bbox["north"]
    
    # ----- ROAD MASK GENERATION -----
    # Create figure with proper size
    fig = plt.figure(
        figsize=(img_size / 100, img_size / 100), dpi=100, frameon=True
    )
    ax = fig.add_axes([0, 0, 1, 1])  # Fill the entire figure

    # Set black background explicitly
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.axis("off")  # Turn off axis

    # Set limits and aspect ratio
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)

    # Adjust figure position to ensure we fill the entire image
    ax.set_position([0, 0, 1, 1])  # Make sure the axes fill the entire figure

    # Plot the road network with white lines
    for u, v, data in g.edges(data=True):
        if "geometry" in data:
            xs, ys = data["geometry"].xy
            ax.plot(
                xs, ys, color="white", linewidth=3, solid_capstyle="round", zorder=1
            )
        else:
            # If no geometry attribute, use node coordinates
            x1 = g.nodes[u]["x"]
            y1 = g.nodes[u]["y"]
            x2 = g.nodes[v]["x"]
            y2 = g.nodes[v]["y"]
            ax.plot(
                [x1, x2],
                [y1, y2],
                color="white",
                linewidth=3,
                solid_capstyle="round",
                zorder=1,
            )

    # Render the figure to an array
    fig.canvas.draw()
    # Use buffer_rgba() instead of deprecated tostring_rgb()
    
    buf = fig.canvas.buffer_rgba()
    rgba_array = np.asarray(buf, dtype=np.uint8)
    # Reshape to proper dimensions with 4 channels (RGBA)
    rgba_array = rgba_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))

    road_mask_img = rgba_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))

    # Ensure we have a 256x256 image by resizing if needed
    if road_mask_img.shape[0] != img_size or road_mask_img.shape[1] != img_size:
        if has_cv2:
            road_mask_img = cv2.resize(road_mask_img, (img_size, img_size))
        else:
            road_mask_img = np.array(Image.fromarray(road_mask_img).resize((img_size, img_size)))
            
    # Create binary mask
    if has_cv2:
        road_mask_gray = cv2.cvtColor(road_mask_img, cv2.COLOR_RGB2GRAY)
        _, road_mask_single_channel = cv2.threshold(
            road_mask_gray, 127, 255, cv2.THRESH_BINARY
        )
        # Double-check for proper black background
        if np.mean(road_mask_single_channel) > 200:  # If mostly white
            road_mask_single_channel = 255 - road_mask_single_channel  # Invert
    else:
        road_mask_gray = np.mean(road_mask_img, axis=2).astype(np.uint8)
        road_mask_single_channel = np.where(road_mask_gray > 127, 255, 0).astype(
            np.uint8
        )
        # Double-check for proper black background
        if np.mean(road_mask_single_channel) > 200:  # If mostly white
            road_mask_single_channel = 255 - road_mask_single_channel  # Invert

    # Save road mask if path is provided
    if output_path:
        if has_cv2:
            cv2.imwrite(str(output_path), road_mask_single_channel)
        else:
            Image.fromarray(road_mask_single_channel).save(str(output_path))

    plt.close(fig)

    # ----- KEYPOINTS VISUALIZATION -----
    # Create figure with proper size
    fig = plt.figure(
        figsize=(img_size / 100, img_size / 100), dpi=100, frameon=True
    )
    ax = fig.add_axes([0, 0, 1, 1])  # Fill the entire figure

    # Set black background explicitly
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.axis("off")  # Turn off axis

    # Set limits and aspect ratio
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    
    # Adjust figure position to ensure we fill the entire image
    ax.set_position([0, 0, 1, 1])  # Make sure the axes fill the entire figure

    # Plot nodes as white points
    for node_id, node_data in graph_data["nodes"].items():
        ax.plot(
            node_data["geo_coords"][0],
            node_data["geo_coords"][1],
            "o",
            color="white",
            markersize=3,
            alpha=1.0,
            zorder=2,
        )

    # Save keypoints if path is provided
    if keypoints_file:
        plt.savefig(keypoints_file, dpi=100, bbox_inches="tight", pad_inches=0)

    # Render the figure to an array
    fig.canvas.draw()
    # Use buffer_rgba() instead of deprecated tostring_rgb()
    buf = fig.canvas.buffer_rgba()
    rgba_array = np.asarray(buf, dtype=np.uint8)
    # Reshape to proper dimensions with 4 channels (RGBA)
    rgba_array = rgba_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    # Convert RGBA to RGB by dropping the alpha channel
    keypoints_img = rgba_array[:, :, :3]
    
    # Ensure we have a 256x256 image by resizing if needed
    if keypoints_img.shape[0] != img_size or keypoints_img.shape[1] != img_size:
        if has_cv2:
            keypoints_img = cv2.resize(keypoints_img, (img_size, img_size))
        else:
            keypoints_img = np.array(Image.fromarray(keypoints_img).resize((img_size, img_size)))

    plt.close(fig)
    
    return road_mask_single_channel, keypoints_img


async def create_road_mask_async(
    bbox: Dict[str, float],
    output_path: Optional[Union[str, Path]] = None,
    keypoints_file: Optional[Union[str, Path]] = None,
    json_file: Optional[Union[str, Path]] = None,
    keep_one_node_in_every: int = 3,
    project_to_osgb: bool = True
) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], Optional[Dict[str, Any]]]:
    """
    Async version of create_road_mask to run in a separate thread to prevent blocking.
    Creates a road mask image using OSMnx for the given bounding box.
    Also saves the graph structure and node coordinates in pixel space (256x256).

    Args:
        bbox: Dictionary containing 'north', 'south', 'east', 'west' coordinates
        output_path: Path where the road mask should be saved (optional)
        keypoints_file: Path where the keypoints visualization should be saved (optional)
        json_file: Path where the graph data should be saved (optional)
        keep_one_node_in_every: Number of intermediate nodes to skip
        project_to_osgb: Whether to project coordinates to EPSG:27700 (British National Grid)

    Returns:
        tuple: (success, road_mask_img, keypoints_img, graph_data)
            - success: Boolean indicating if the operation was successful
            - road_mask_img: NumPy array of the road mask image (single channel)
            - keypoints_img: NumPy array of the keypoints visualization (RGB)
            - graph_data: Dictionary containing the graph structure
    """
    # Run the synchronous function in a thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        create_road_mask,
        bbox,
        output_path,
        keypoints_file,
        json_file,
        keep_one_node_in_every,
        project_to_osgb,
    )


async def download_uk_tiles_async(
    config: Dict[str, Any], dataset_handler: Optional[DatasetHandler] = None
) -> None:
    """
    Asynchronously download tiles in a grid pattern within the specified bounds.
    Images are automatically converted to PNG format for consistency.
    Includes rate limiting and retry with backoff.

    Args:
        config: Dictionary containing all configuration parameters including:
            - url: Base URL for the tile service
            - map_id: ID of the map tileset
            - zoom: Zoom level
            - format: Image format (jpg/png) for the api only
            - output_dir: Output directory for tiles
            - bounds: Dictionary with 'west', 'south', 'east', 'north' coordinates
            - grid_size: Tuple of (rows, cols) for sampling grid
            - rate_limit: Maximum requests per second (default: 5)
            - max_retries: Maximum retry attempts (default: 5)
        dataset_handler: Optional pre-initialized dataset handler. If None, a new one will be created.
    """
    # Check if cv2 methods are available
    has_cv2 = all(
        hasattr(cv2, attr)
        for attr in ["imdecode", "IMREAD_COLOR", "imwrite", "IMWRITE_PNG_COMPRESSION"]
    )

    # Validate and get bounds from config
    bounds = config.get("bounds")
    if not bounds or not all(k in bounds for k in ["west", "south", "east", "north"]):
        raise ValueError(
            "Config must include valid bounds with west, south, east, and north coordinates"
        )

    url = config["url"]
    map_id = config["map_id"]
    zoom = config.get("zoom", 10)
    format = config.get("format", "png")
    output_dir = config.get("output_dir", "src/extractors/geolocator/dataset/tiles")
    grid_size = config.get("grid_size", (10, 10))

    # Rate limiting and retry configuration
    rate_limit = config.get("rate_limit", 5)  # requests per second
    max_retries = config.get("max_retries", 5)
    max_concurrent = config.get("max_concurrent", 5)  # max concurrent requests

    # Initialize rate limiter and retry handler
    rate_limiter = RateLimiter(rate=rate_limit, burst=max_concurrent)
    retry_handler = RetryWithBackoff(
        max_retries=max_retries,
        retry_on_exceptions=(
            aiohttp.ClientError,
            asyncio.TimeoutError,
            requests.RequestException,
        ),
    )

    # Initialize dataset handler if not provided
    if dataset_handler is None:
        dataset_handler = DatasetHandler(output_dir)

    # Convert bounds to tile coordinates
    min_x, max_y = deg2num(bounds["south"], bounds["west"], zoom)
    max_x, min_y = deg2num(bounds["north"], bounds["east"], zoom)

    # Calculate tile spans
    x_span = max_x - min_x
    y_span = max_y - min_y

    # Calculate step sizes for grid
    rows, cols = grid_size
    x_step = x_span / (cols - 1) if cols > 1 else 0
    y_step = y_span / (rows - 1) if rows > 1 else 0

    # Ensure steps are at least 1 tile width to prevent overlapping
    if x_step < 1 or y_step < 1:
        raise ValueError(
            f"Grid size {grid_size} is too large for the given bounds. Steps must be at least 1 tile width. Current steps: x={x_step:.2f}, y={y_step:.2f}"
        )

    total_tiles = rows * cols
    sample_index = 0

    # Async function to download a single tile
    async def download_single_tile(i, j, sample_idx):
        nonlocal sample_index

        # Calculate tile coordinates for this grid position
        x = int(min_x + (j * x_step))
        y = int(min_y + (i * y_step))

        # Convert tile coordinates back to lat/lon for center of tile
        lat, lon = num2deg(x + 0.5, y + 0.5, zoom)

        tile_id = f"{sample_idx:05d}_{lat:.5f}_{lon:.5f}"

        # Get tile metadata
        bbox = get_tile_bbox(x, y, zoom)
        scale = get_tile_scale(lat, zoom)

        metadata = {
            "center": {"lat": lat, "lon": lon},
            "bbox": bbox,
            "scale": scale,
            "zoom": zoom,
            "tile_coordinates": {"x": x, "y": y},
            "grid_position": {"row": i, "col": j},
            "map_id": map_id,
            "map_group": config.get("map_group", ""),
            "source_scale": config.get("scale", 0),
        }

        # Check if this tile already exists in our dataset
        try:
            dataset_handler.get_tile_data(tile_id)
            logger.debug(f"Tile {tile_id} already exists, skipping...")
            return
        except ValueError:
            # Tile doesn't exist, proceed with download
            pass

        # Wait for rate limiter before making the request
        await rate_limiter.acquire()

        # Format URL according to the NLS style: url/{z}/{x}/{y}.png
        tile_url = f"{url}/{zoom}/{x}/{y}.{format}"

        async def fetch_tile():
            async with aiohttp.ClientSession() as session:
                async with session.get(tile_url, timeout=30) as response:
                    if response.status != 200:
                        raise aiohttp.ClientResponseError(
                            request_info=response.request_info,
                            history=response.history,
                            status=response.status,
                            message=f"HTTP error {response.status}",
                            headers=response.headers,
                        )
                    return await response.read()

        try:
            # Use retry handler to fetch tile with backoff
            content = await retry_handler.execute(fetch_tile)

            # Create road mask (this runs in a thread pool to avoid blocking)
            (
                success,
                road_mask_img,
                keypoints_img,
                graph_data,
            ) = await create_road_mask_async(
                bbox,
                keep_one_node_in_every=config.get("keep_one_node_in_every", 3),
            )

            if success:
                logger.info(f"Created road mask for grid tile [{i},{j}]")

                # Update metadata with graph data
                metadata["graph_data"] = {
                    "node_count": len(graph_data["nodes"]),
                    "edge_count": len(graph_data["edges"]),
                }
                
                # Add projection information to metadata
                metadata["projection"] = {
                    "original": "EPSG:4326",  # WGS84
                    "projected": "EPSG:27700", # British National Grid
                }

                # Convert image data based on availability of OpenCV
                if has_cv2:
                    # OpenCV based conversion
                    image_array = np.frombuffer(content, np.uint8)
                    raster_img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                    # Convert BGR to RGB if needed
                    if len(raster_img.shape) == 3 and raster_img.shape[2] == 3:
                        raster_img = raster_img[:, :, ::-1]  # BGR to RGB
                else:
                    # PIL based conversion
                    import io

                    img = Image.open(io.BytesIO(content))
                    raster_img = np.array(img)

                # Save all data for this tile using the dataset handler
                dataset_handler.save_tile_data(
                    tile_id=tile_id,
                    raster_img=raster_img,
                    road_mask_img=road_mask_img,
                    keypoints_img=keypoints_img,
                    graph_data=graph_data,
                    metadata=metadata,
                    raster_bytes=content,
                )

                logger.info(
                    f"Downloaded grid tile [{i},{j}] at {x},{y} ({sample_idx + 1}/{total_tiles})"
                )
            else:
                logger.warning(f"Failed to create road mask for grid tile [{i},{j}]")
        except Exception as e:
            logger.error(f"Error downloading grid tile [{i},{j}] at {x},{y}: {e}")

    # Create tasks for each tile
    tasks = []
    for i in range(rows):
        for j in range(cols):
            tasks.append(download_single_tile(i, j, sample_index))
            sample_index += 1

    # Create a semaphore to limit concurrent tasks
    semaphore = asyncio.Semaphore(max_concurrent)

    # Wrap each task with the semaphore
    async def bounded_download(coro):
        async with semaphore:
            return await coro

    # Execute all tasks with concurrency control
    await asyncio.gather(*(bounded_download(task) for task in tasks))

    # Print dataset statistics
    logger.info("\n=== Dataset Statistics ===")
    stats = dataset_handler.get_dataset_stats()
    for key, value in stats.items():
        logger.info(f"{key}: {value}")

    # Generate train/val/test split if configured
    if config.get("generate_split", True):
        logger.info("\n=== Generating train/validation/test split ===")
        logger.info(
            f"Total tiles in dataset: {len(dataset_handler.get_all_tile_ids())}"
        )
        dataset_handler.generate_data_split(
            train_ratio=config.get("train_ratio", 0.8),
            val_ratio=config.get("val_ratio", 0.1),
            test_ratio=config.get("test_ratio", 0.1),
            seed=config.get("split_seed", 42),
        )

        # Upload the split file to GCS if it's enabled
        if dataset_handler.upload_to_gcs_enabled:
            split_file = dataset_handler.dataset_dir / "data_split.json"
            gcs_url = dataset_handler.upload_file_to_gcs(split_file)
            if gcs_url:
                logger.info(f"Uploaded data split to {gcs_url}")

    # Log GCS location if uploads were enabled
    if dataset_handler.upload_to_gcs_enabled:
        gcs_bucket = config.get("gcs_bucket", "extract-general")
        destination_prefix = config.get(
            "destination_prefix", f"sam_road/datasets/{dataset_handler.dataset_id}"
        )
        logger.info(f"Dataset available at: gs://{gcs_bucket}/{destination_prefix}")

    # Verify and sync all GCS uploads to make sure everything is properly uploaded
    if dataset_handler.upload_to_gcs_enabled:
        logger.info("Starting final verification of all GCS uploads...")
        upload_stats = dataset_handler.verify_and_sync_gcs_uploads()
        logger.info(f"GCS upload verification complete: {upload_stats}")

        # Check if there were any failures
        if upload_stats.get("failed_uploads", 0) > 0:
            logger.warning(
                f"WARNING: {upload_stats['failed_uploads']} files failed to upload to GCS. "
                f"Manual intervention may be required."
            )
        else:
            logger.info("All files successfully verified and uploaded to GCS!")


async def main_async():
    try:
        # Map data
        map_data = [
            {
                "scale": 2500,
                "url": "https://geo.nls.uk/mapdata2/os/2500_A_1D/",
                "map_id": "os_2500_A_1D",
                "map_group": "OS 1:2,500 A edition England / Wales, 1 x 2 km sheets, 1944-1973",
            },
            {
                "scale": 2500,
                "url": "https://geo.nls.uk/mapdata2/os/2500_A_2D/",
                "map_id": "os_2500_A_2D",
                "map_group": "OS 1:2,500 A edition England / Wales, 1 x 2 km sheets, 1944-1973",
            },
            {
                "scale": 2500,
                "url": "https://geo.nls.uk/mapdata2/os/2500_A_3D/",
                "map_id": "os_2500_A_3D",
                "map_group": "OS 1:2,500 A edition England / Wales, 1 x 2 km sheets, 1944-1973",
            },
            {
                "scale": 2500,
                "url": "https://geo.nls.uk/mapdata2/os/2500_A_4D/",
                "map_id": "os_2500_A_4D",
                "map_group": "OS 1:2,500 A edition England / Wales, 1 x 2 km sheets, 1944-1973",
            },
            {
                "scale": 2500,
                "url": "https://geo.nls.uk/mapdata2/os/2500_A_5D/",
                "map_id": "os_2500_A_5D",
                "map_group": "OS 1:2,500 A edition England / Wales, 1 x 2 km sheets, 1944-1973",
            },
            {
                "scale": 2500,
                "url": "https://geo.nls.uk/mapdata2/os/2500_A_6D/",
                "map_id": "os_2500_A_6D",
                "map_group": "OS 1:2,500 A edition England / Wales, 1 x 2 km sheets, 1944-1973",
            },
            {
                "scale": 2500,
                "url": "https://geo.nls.uk/mapdata2/os/2500_A_7D/",
                "map_id": "os_2500_A_7D",
                "map_group": "OS 1:2,500 A edition England / Wales, 1 x 2 km sheets, 1944-1973",
            },
        ]

        # Common configuration for all datasets
        base_config = {
            "zoom": 16,
            "format": "png",
            "bounds": {
                "west": -2.796278,
                "south": 51.088321,
                "east": -1.055098,
                "north": 54.486221,
            },
            "grid_size": (120, 30),  # Smaller grid for testing
            "extra_nodes": True,
            "keep_one_node_in_every": 5,
            # GCS upload configuration - enabled by default
            "upload_to_gcs": True,
            "gcs_bucket": "extract-general",
            # Data split configuration
            "generate_split": True,
            "train_ratio": 0.8,
            "val_ratio": 0.1,
            "test_ratio": 0.1,
            "split_seed": 42,
            # New async-specific settings
            "rate_limit": 1000,  # requests per second
            "max_retries": 5,  # maximum retry attempts
            "max_concurrent": 30,  # maximum concurrent requests
        }

        # Process each map source
        for map_source in map_data:
            logger.info(f"\n=== Processing map source: {map_source['map_id']} ===")

            # Create a unique dataset ID for this map source
            dataset_name = map_source["map_id"]
            dataset_id = f"{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M')}"

            # Combine base config with map-specific data
            config = base_config.copy()
            config.update(map_source)
            config["dataset_name"] = dataset_name
            config["dataset_id"] = dataset_id
            config["output_dir"] = f"src/extractors/georeferencers/dataset/{dataset_id}"

            # Initialize the dataset handler
            dataset_handler = DatasetHandler(
                config["output_dir"], dataset_id=dataset_id
            )

            # Enable GCS uploads if configured
            if config.get("upload_to_gcs", True) and config.get("gcs_bucket"):
                logger.info(
                    f"Enabling incremental uploads to GCS bucket {config['gcs_bucket']}"
                )
                destination_prefix = f"sam_road/datasets/{dataset_id}"
                config["destination_prefix"] = destination_prefix
                dataset_handler.enable_gcs_upload(
                    config["gcs_bucket"], destination_prefix
                )

            # Save config to dataset metadata
            dataset_handler.dataset_index["metadata"]["config"] = config
            dataset_handler.dataset_index["metadata"]["map_source"] = map_source
            dataset_handler._save_index()

            # Download tiles asynchronously for this map source
            logger.info(f"Starting download for {dataset_id}")
            await download_uk_tiles_async(config, dataset_handler=dataset_handler)
            logger.info(f"Finished processing {dataset_id}")

        logger.info("All map sources processed successfully")

    except ValueError as e:
        logger.error(f"Error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback

        logger.error(traceback.format_exc())


def main():
    asyncio.run(main_async())


def create_composite_dataset(
    dataset_ids: List[str],
    output_dir: Optional[Union[str, Path]] = None,
    composite_name: Optional[str] = None,
    upload_to_gcs: bool = False,
    gcs_bucket: str = "extract-general",
    gcs_base_path: Optional[str] = None,
) -> DatasetHandler:
    """
    Create a composite dataset by combining multiple existing datasets.
    The composite dataset preserves the original train/test/validation splits.

    Args:
        dataset_ids: List of dataset IDs to combine
        output_dir: Directory to store the composite dataset (if None, uses a timestamped directory)
        composite_name: Name for the composite dataset (if None, auto-generated)
        upload_to_gcs: Whether to upload the composite dataset to GCS
        gcs_bucket: GCS bucket name for upload
        gcs_base_path: Custom base path in GCS bucket (if None, uses standard path based on dataset_id)

    Returns:
        DatasetHandler for the new composite dataset
    """
    # Generate a timestamp for uniqueness
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    # Create a name for the composite dataset if not provided
    if not composite_name:
        composite_name = f"composite_{timestamp}"

    # Create output directory if not provided
    if not output_dir:
        output_dir = f"src/extractors/georeferencers/dataset/{composite_name}"

    # Create the composite dataset handler
    composite_handler = DatasetHandler(output_dir, dataset_id=composite_name)

    # Keep track of all source datasets and their splits
    combined_splits = {"train": [], "validation": [], "test": []}
    all_source_metadata = []
    tile_mapping = {}  # Maps original tile IDs to new tile IDs

    # Track dataset source per split
    split_source_distribution = {"train": {}, "validation": {}, "test": {}}

    logger.info(f"Creating composite dataset from {len(dataset_ids)} source datasets")

    # Process each source dataset
    for idx, dataset_id in enumerate(dataset_ids):
        logger.info(
            f"Processing source dataset {idx+1}/{len(dataset_ids)}: {dataset_id}"
        )

        # Initialize source count in distribution for each split
        for split in split_source_distribution:
            split_source_distribution[split][dataset_id] = 0

        # Try different paths to find the dataset
        potential_paths = [
            f"src/extractors/georeferencers/dataset/{dataset_id}",
            dataset_id,  # In case a full path is provided
        ]

        source_handler = None
        for path in potential_paths:
            try:
                source_handler = DatasetHandler(
                    path, download_from_gcs=True, dataset_id=dataset_id
                )
                break
            except Exception as e:
                logger.debug(f"Failed to load dataset from {path}: {e}")
                continue

        if not source_handler:
            logger.warning(f"Could not load dataset: {dataset_id}, skipping")
            continue

        # Get source dataset metadata and splits
        source_metadata = source_handler.get_dataset_stats()
        source_metadata["dataset_id"] = dataset_id
        source_metadata["source_index"] = idx
        all_source_metadata.append(source_metadata)

        # Get the data split for this source dataset
        source_split = source_handler.get_data_split()
        if not source_split:
            logger.warning(
                f"Dataset {dataset_id} has no train/test/val split defined. All tiles will be added to the training set."
            )
            source_split = {
                "train": source_handler.get_all_tile_ids(),
                "validation": [],
                "test": [],
            }

        # Copy each tile from source dataset to the composite dataset
        all_tile_ids = source_handler.get_all_tile_ids()
        for i, tile_id in enumerate(all_tile_ids):
            if i % 100 == 0:
                logger.info(
                    f"Copying tile {i+1}/{len(all_tile_ids)} from dataset {dataset_id}"
                )

            # Create a new unique ID for this tile in the composite dataset
            new_tile_id = f"{dataset_id}_{tile_id}"
            tile_mapping[f"{dataset_id}:{tile_id}"] = new_tile_id

            try:
                # Get tile data from source
                tile_data = source_handler.get_tile_data(tile_id)

                # Add source dataset info to tile metadata
                if "metadata" not in tile_data:
                    tile_data["metadata"] = {}

                tile_data["metadata"]["source_dataset"] = dataset_id
                tile_data["metadata"]["original_tile_id"] = tile_id

                # Save the tile in the composite dataset
                composite_handler.save_tile_data(
                    tile_id=new_tile_id,
                    raster_img=tile_data.get("raster_img"),
                    road_mask_img=tile_data.get("road_mask_img"),
                    keypoints_img=tile_data.get("keypoints_img"),
                    graph_data=tile_data.get("graph_data"),
                    metadata=tile_data.get("metadata"),
                )

                # Add to the appropriate split based on the source dataset's split
                split_category = None
                if tile_id in source_split["train"]:
                    combined_splits["train"].append(new_tile_id)
                    split_category = "train"
                elif tile_id in source_split["validation"]:
                    combined_splits["validation"].append(new_tile_id)
                    split_category = "validation"
                elif tile_id in source_split["test"]:
                    combined_splits["test"].append(new_tile_id)
                    split_category = "test"
                else:
                    # Default to train if not in any split
                    combined_splits["train"].append(new_tile_id)
                    split_category = "train"

                # Update split source distribution
                split_source_distribution[split_category][dataset_id] += 1

            except Exception as e:
                logger.error(
                    f"Error copying tile {tile_id} from dataset {dataset_id}: {e}"
                )

    # Calculate percentages for each split
    split_source_distribution_pct = {
        split_name: {} for split_name in split_source_distribution.keys()
    }

    for split_name, sources in split_source_distribution.items():
        total_in_split = sum(sources.values())
        if total_in_split > 0:
            for source_id, count in sources.items():
                split_source_distribution_pct[split_name][source_id] = (
                    count / total_in_split
                ) * 100

    # Save the combined splits to the composite dataset
    split_file = Path(composite_handler.dataset_dir) / "data_split.json"
    with open(split_file, "w") as f:
        json.dump(combined_splits, f, indent=2)

    # Calculate overall dataset source distribution
    overall_source_distribution = {}
    total_tiles = sum(len(split) for split in combined_splits.values())

    # Combine counts from all splits for each source
    for split_name, sources in split_source_distribution.items():
        for source_id, count in sources.items():
            if source_id not in overall_source_distribution:
                overall_source_distribution[source_id] = 0
            overall_source_distribution[source_id] += count

    # Calculate percentages
    overall_source_distribution_pct = {}
    for source_id, count in overall_source_distribution.items():
        overall_source_distribution_pct[source_id] = (count / total_tiles) * 100

    # Sort distributions by percentage (descending)
    sorted_distribution = {
        k: v
        for k, v in sorted(
            overall_source_distribution_pct.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    }

    # Update composite dataset metadata
    composite_handler.dataset_index["metadata"][
        "is_composite"
    ] = True  # Set the composite flag
    composite_handler.dataset_index["metadata"]["composite"] = {
        "created_at": datetime.now().isoformat(),
        "source_datasets": dataset_ids,
        "source_metadata": all_source_metadata,
        "tile_mapping": tile_mapping,
        "combined_split": {
            "train_count": len(combined_splits["train"]),
            "validation_count": len(combined_splits["validation"]),
            "test_count": len(combined_splits["test"]),
            "total_count": sum(len(split) for split in combined_splits.values()),
        },
        # Add simple split source distribution
        "split_source_distribution": {
            "counts": split_source_distribution,
            "percentages": split_source_distribution_pct,
        },
        # Add overall dataset source distribution
        "overall_source_distribution": {
            "counts": overall_source_distribution,
            "percentages": overall_source_distribution_pct,
            "sorted_percentages": sorted_distribution,
        },
    }

    # Save the updated index
    composite_handler._save_index()

    # Upload to GCS if requested
    if upload_to_gcs:
        logger.info(f"Enabling GCS uploads to bucket {gcs_bucket}")

        try:
            # Enable GCS uploads
            base_path = gcs_base_path or f"sam_road/datasets/{composite_name}"
            composite_handler.enable_gcs_upload(gcs_bucket, base_path)

            # Upload the split file to GCS
            gcs_url = composite_handler.upload_file_to_gcs(split_file)
            if gcs_url:
                logger.info(f"Uploaded data split to {gcs_url}")

            # Verify all files are uploaded
            logger.info("Starting verification of all GCS uploads...")
            upload_stats = composite_handler.verify_and_sync_gcs_uploads()
            logger.info(f"GCS upload verification complete: {upload_stats}")

            # Check if there were any failures
            if upload_stats.get("failed_uploads", 0) > 0:
                logger.warning(
                    f"WARNING: {upload_stats['failed_uploads']} files failed to upload to GCS. "
                    f"Manual intervention may be required."
                )
            else:
                logger.info("All files successfully verified and uploaded to GCS!")
                logger.info(
                    f"Composite dataset available at: gs://{gcs_bucket}/{base_path}"
                )

        except Exception as e:
            logger.error(f"Error uploading to GCS: {e}")

    # Log split source distribution
    logger.info("\n=== Split Source Distribution ===")
    for split_name, sources in split_source_distribution.items():
        logger.info(
            f"\n{split_name.capitalize()} Split ({sum(sources.values())} tiles):"
        )
        for source_id, count in sources.items():
            percentage = split_source_distribution_pct[split_name].get(source_id, 0)
            logger.info(f"  {source_id}: {count} tiles ({percentage:.1f}%)")

    logger.info(
        f"\nComposite dataset created successfully with {composite_handler.get_dataset_stats()['tile_count']} tiles"
    )
    logger.info(
        f"Train: {len(combined_splits['train'])}, Validation: {len(combined_splits['validation'])}, Test: {len(combined_splits['test'])}"
    )

    return composite_handler


if __name__ == "__main__":
    create_composite_dataset(
        dataset_ids=[
            "os_2500_A_1D_20250308_1525",
            "os_2500_A_2D_20250308_1630",
            "os_2500_A_3D_20250308_1736",
            "os_2500_A_4D_20250308_1842",
            "os_2500_A_5D_20250308_1942",
            "os_2500_A_6D_20250308_2046",
            "os_sparse_20250307_1717",
        ],
        composite_name="os_composite_bw_color",
        upload_to_gcs=True,
        gcs_bucket="extract-general",
    )