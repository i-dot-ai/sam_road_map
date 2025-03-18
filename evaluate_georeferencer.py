#!/usr/bin/env python3
"""
Evaluate the RoadMatcherGeoreferencer against a dataset.

## evaluate_georeferencer.py

This script evaluates the accuracy of the RoadMatcherGeoreferencer by comparing original points with georeferenced points.

### Usage

```bash
python eval/evaluate_georeferencer.py [OPTIONS]
```

### Options

- `--config PATH`: Path to the configuration file (default: "config/toponet_vitb_256_os.yaml")
- `--output_dir DIR`: Directory to save output files (default: "georef_eval_TIMESTAMP")
- `--num_samples N`: Number of samples to process (default: all)
- `--zoom LEVEL`: Zoom level for georeferencing (default: 16)
- `--no_debug`: Disable debug mode

### Output

The script generates the following outputs:

1. A markdown report containing statistics and visualizations
2. JSON results file with detailed statistics
3. CSV file with evaluation data for further analysis
4. Visualizations:
   - Histogram of error distances
   - Scatter plot of offset distance vs. error
   - Cumulative distribution function (CDF) of errors
5. Individual sample results in separate folders

### Example

```bash
# Evaluate with default settings
python eval/evaluate_georeferencer.py

# Evaluate a specific number of samples with custom output directory
python eval/evaluate_georeferencer.py --num_samples 100 --output_dir my_evaluation

# Evaluate with a different zoom level and without debug mode
python eval/evaluate_georeferencer.py --zoom 17 --no_debug
```

### Metrics Reported

- Mean, median, min, max error distances
- Standard deviation of errors
- Percentage of points within 10m, 25m, 50m, and 100m
- Correlation between offset distance and error 
"""

import os
import math
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from shapely.geometry import Point
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
import pandas as pd
from pathlib import Path

from extract.extractors.georeferencers import RoadMatcherGeoreferencer
from dataset import SatMapDataset
from utils import load_config, create_output_dir_and_save_config

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def haversine_distance(point1, point2):
    """
    Calculate the distance between two points in decimal degrees
    using the Haversine formula for more accurate geospatial distance.
    
    Args:
        point1 (tuple): (lon, lat) of first point
        point2 (tuple): (lon, lat) of second point
    
    Returns:
        float: Distance in meters
    """
    # Convert decimal degrees to radians
    lon1, lat1 = point1
    lon2, lat2 = point2
    
    # Radius of earth in meters
    R = 6371000.0
    
    # Haversine formula
    dlon = math.radians(lon2 - lon1)
    dlat = math.radians(lat2 - lat1)
    a = (math.sin(dlat/2)**2 + 
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
         math.sin(dlon/2)**2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c
    
    return distance

def evaluate_georeferencer(config, output_dir, zoom=16, num_samples=None, debug_mode=True):
    """
    Evaluate the georeferencer against a dataset.
    
    Args:
        config: Configuration object
        output_dir: Directory to save output files
        zoom: Zoom level for georeferencing (default: 16)
        num_samples: Number of samples to process (default: None, process all)
        debug_mode: Whether to enable debug mode (default: True)
    
    Returns:
        dict: Evaluation results
    """
    # Initialize dataset
    logger.info("Initializing dataset...")
    val_ds = SatMapDataset(config, is_train=False, return_graph=False, return_metadata=True)
    
    # Initialize georeferencer
    logger.info("Initializing georeferencer...")
    georeferencer = RoadMatcherGeoreferencer(sam_config_path=config.__config_path__, debug_mode=debug_mode,debug_output_dir=output_dir)
    
    # Create lists to store data for analysis
    original_points = []
    offset_points = []
    new_points = []
    distances = []
    offset_distances = []
    sample_data = []
    
    # Set number of samples to process
    total_samples = len(val_ds) if num_samples is None else min(num_samples, len(val_ds))
    
    # Process samples
    logger.info(f"Processing {total_samples} samples...")
    for i in tqdm(range(total_samples)):
        try:
            # Get sample
            datum = val_ds[i]
            metadata = datum['metadata']
            
            # Extract original center point
            original_lat = metadata['center']['lat']
            original_lon = metadata['center']['lon']
            original_point = (original_lon, original_lat)
            
            # Extract offset point
            offset_lat = metadata['offset']['lat']
            offset_lon = metadata['offset']['lon']
            offset_point = Point(offset_lon, offset_lat)
            
            # Apply georeferencing to get corrected point
            try:
                new_point = georeferencer.georeference(datum['rgb'], offset_point, zoom=zoom,debug_ouput_subfolder=f"sample_{i}").tolist()
            except ValueError as ve:
                logger.warning(f"ValueError in sample {i} during georeferencing: {str(ve)}")
                # Skip this sample and continue with next one
                continue
            
            # Calculate distance between new point and original point (in meters)
            distance_meters = haversine_distance(original_point, new_point)
            
            # Store data for analysis
            original_points.append(original_point)
            offset_points.append((offset_lon, offset_lat))
            new_points.append(new_point)
            distances.append(distance_meters)
            offset_distances.append(metadata['offset']['distance'])
            
            # Create sample data dictionary
            sample_info = {
                "sample_id": i,
                "original_point": original_point,
                "offset_point": (offset_lon, offset_lat),
                "new_point": new_point,
                "distance_error": distance_meters,
                "offset_distance": metadata['offset']['distance']
            }
            
            # Save sample data
            sample_data.append(sample_info)
            
            # Save sample-level results
            sample_dir = os.path.join(output_dir, f"sample_{i}")
            os.makedirs(sample_dir, exist_ok=True)
            with open(os.path.join(sample_dir, "results.json"), "w") as f:
                json.dump(sample_info, f, indent=4)
            
        except Exception as e:
            logger.error(f"Error processing sample {i}: {str(e)}")
            continue
    
    # Convert to numpy arrays for analysis
    distances = np.array(distances)
    offset_distances = np.array(offset_distances)
    
    # Calculate statistics
    stats = {
        "num_samples": len(distances),
        "mean_error": float(np.mean(distances)),
        "median_error": float(np.median(distances)),
        "min_error": float(np.min(distances)),
        "max_error": float(np.max(distances)),
        "std_error": float(np.std(distances)),
        "within_10m_percent": float(np.sum(distances < 10) / len(distances) * 100),
        "within_25m_percent": float(np.sum(distances < 25) / len(distances) * 100),
        "within_50m_percent": float(np.sum(distances < 50) / len(distances) * 100),
        "within_100m_percent": float(np.sum(distances < 100) / len(distances) * 100),
        "mean_offset_distance": float(np.mean(offset_distances)),
        "median_offset_distance": float(np.median(offset_distances))
    }
    
    # Create dataframe for further analysis
    df = pd.DataFrame({
        "sample_id": range(len(distances)),
        "distance_error": distances,
        "offset_distance": offset_distances
    })
    
    # Calculate correlation between offset distance and error
    correlation = np.corrcoef(offset_distances, distances)[0, 1]
    stats["offset_error_correlation"] = float(correlation)
    
    # Return evaluation results
    results = {
        "statistics": stats,
        "samples": sample_data
    }
    
    return results, df

def generate_visualizations(results, df, output_dir):
    """
    Generate visualizations from evaluation results.
    
    Args:
        results: Evaluation results
        df: DataFrame with evaluation data
        output_dir: Directory to save output files
    """
    stats = results["statistics"]
    distances = df["distance_error"].values
    
    # Create output directory for visualizations
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Create a histogram of distances
    plt.figure(figsize=(10, 6))
    plt.hist(distances, bins=20, alpha=0.7)
    plt.axvline(stats["median_error"], color='r', linestyle='dashed', linewidth=1, 
                label=f'Median: {stats["median_error"]:.2f}m')
    plt.axvline(stats["mean_error"], color='g', linestyle='dashed', linewidth=1, 
                label=f'Mean: {stats["mean_error"]:.2f}m')
    plt.xlabel('Error Distance (meters)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Georeferencing Errors')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(vis_dir, "error_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a scatter plot of offset distance vs error
    plt.figure(figsize=(10, 6))
    plt.scatter(df["offset_distance"], df["distance_error"], alpha=0.5)
    plt.xlabel('Offset Distance (meters)')
    plt.ylabel('Error Distance (meters)')
    plt.title(f'Offset Distance vs. Error (Correlation: {stats["offset_error_correlation"]:.3f})')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(vis_dir, "offset_vs_error.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a CDF of errors
    plt.figure(figsize=(10, 6))
    sorted_distances = np.sort(distances)
    cumulative = np.arange(1, len(sorted_distances) + 1) / len(sorted_distances)
    plt.plot(sorted_distances, cumulative)
    plt.grid(True, alpha=0.3)
    plt.xlabel('Error Distance (meters)')
    plt.ylabel('Cumulative Probability')
    plt.title('Cumulative Distribution Function of Georeferencing Errors')
    
    # Add markers for specific percentiles
    for p, color, label in [(0.5, 'r', f'50% < {np.median(distances):.1f}m'), 
                           (0.75, 'g', f'75% < {np.percentile(distances, 75):.1f}m'),
                           (0.9, 'b', f'90% < {np.percentile(distances, 90):.1f}m')]:
        idx = int(p * len(sorted_distances)) - 1
        plt.plot(sorted_distances[idx], p, marker='o', color=color)
        plt.axhline(p, color=color, linestyle='--', alpha=0.3)
        plt.axvline(sorted_distances[idx], color=color, linestyle='--', alpha=0.3)
        plt.text(sorted_distances[idx] + 5, p, label, va='center')
    
    plt.savefig(os.path.join(vis_dir, "error_cdf.png"), dpi=300, bbox_inches='tight')
    plt.close()

def generate_report(results, output_dir):
    """
    Generate a report from evaluation results.
    
    Args:
        results: Evaluation results
        output_dir: Directory to save output files
    """
    stats = results["statistics"]
    report_file = os.path.join(output_dir, "report.md")
    
    with open(report_file, "w") as f:
        f.write("# Georeferencer Evaluation Report\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Summary Statistics\n\n")
        f.write(f"- Number of samples: {stats['num_samples']}\n")
        f.write(f"- Mean error distance: {stats['mean_error']:.2f} meters\n")
        f.write(f"- Median error distance: {stats['median_error']:.2f} meters\n")
        f.write(f"- Min error distance: {stats['min_error']:.2f} meters\n")
        f.write(f"- Max error distance: {stats['max_error']:.2f} meters\n")
        f.write(f"- Standard deviation: {stats['std_error']:.2f} meters\n")
        f.write(f"- Mean offset distance: {stats['mean_offset_distance']:.2f} meters\n")
        f.write(f"- Correlation between offset and error: {stats['offset_error_correlation']:.3f}\n\n")
        
        f.write("## Accuracy\n\n")
        f.write(f"- Points within 10m: {stats['within_10m_percent']:.1f}%\n")
        f.write(f"- Points within 25m: {stats['within_25m_percent']:.1f}%\n")
        f.write(f"- Points within 50m: {stats['within_50m_percent']:.1f}%\n")
        f.write(f"- Points within 100m: {stats['within_100m_percent']:.1f}%\n\n")
        
        f.write("## Visualizations\n\n")
        f.write("### Error Distribution\n\n")
        f.write("![Error Distribution](visualizations/error_distribution.png)\n\n")
        
        f.write("### Offset vs. Error\n\n")
        f.write("![Offset vs. Error](visualizations/offset_vs_error.png)\n\n")
        
        f.write("### Error CDF\n\n")
        f.write("![Error CDF](visualizations/error_cdf.png)\n\n")
    
    # Also save results as JSON
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)

def main():
    """Main entry point for the script."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate the RoadMatcherGeoreferencer")
    parser.add_argument("--config", type=str, default="config/toponet_vitb_256_os.yaml",
                        help="Path to the configuration file")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save output files (default: georef_eval_TIMESTAMP)")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of samples to process (default: all)")
    parser.add_argument("--zoom", type=int, default=16,
                        help="Zoom level for georeferencing (default: 16)")
    parser.add_argument("--no_debug", action="store_true",
                        help="Disable debug mode")
    args = parser.parse_args()
    
    # Load configuration
    config_path = args.config
    config = load_config(config_path)
    
    # Store the config path with the config object
    config.__config_path__ = config_path
    
    # Create output directory
    output_prefix = "georef_eval"
    output_dir = create_output_dir_and_save_config(output_prefix, config, args.output_dir)
    logger.info(f"Output directory: {output_dir}")
    
    # Evaluate georeferencer
    results, df = evaluate_georeferencer(
        config, output_dir, args.zoom, args.num_samples, not args.no_debug
    )
    
    # Generate visualizations
    generate_visualizations(results, df, output_dir)
    
    # Generate report
    generate_report(results, output_dir)
    
    # Save dataframe
    df.to_csv(os.path.join(output_dir, "evaluation_data.csv"), index=False)
    
    logger.info(f"Evaluation complete. Report saved to: {os.path.join(output_dir, 'report.md')}")

if __name__ == "__main__":
    main() 