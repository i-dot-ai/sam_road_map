import yaml
from addict import Dict
from datetime import datetime
import os
from google.cloud import storage
import logging

def load_config(path):
    with open(path) as file:
        config_dict = yaml.safe_load(file)
        
    return Dict(config_dict)

def load_data_config(path, config):
    # The path should now point to the dataset directory
    index_path = os.path.join(path, 'index.json')
    
    with open(index_path) as file:
        index_data = yaml.safe_load(file)
        
    # Extract configuration from the metadata field
    if 'metadata' in index_data:
        data_config_dict = index_data['metadata']
    else:
        # Fallback to original behavior if metadata not found
        with open(path) as file:
            data_config_dict = yaml.safe_load(file)
            
    return Dict(data_config_dict)

def create_output_dir_and_save_config(output_dir_prefix, config, specified_dir=None):
    if specified_dir:
        output_dir = specified_dir
    else:
        # Generate the output directory name with the current timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{output_dir_prefix}_{timestamp}"
    
    # Create the directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Define the path for the config file
    config_path = os.path.join(output_dir, "config.yaml")
    
    # Save the config as a YAML file
    with open(config_path, 'w') as file:
        yaml.dump(config.to_dict(), file)
    
    return output_dir

def upload_to_gcs_bucket(source_directory, bucket_name, destination_prefix=None):
    """
    Uploads the entire content of a directory to a Google Cloud Storage bucket.
    
    Args:
        source_directory (str): Local directory to upload
        bucket_name (str): Name of the GCS bucket
        destination_prefix (str, optional): Prefix for the destination in the bucket
                                           (essentially a "directory" in the bucket)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Initialize GCS client
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        # Log the upload beginning
        logging.info(f"Starting upload of {source_directory} to GCS bucket {bucket_name}")
        
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
                logging.info(f"Uploaded {local_path} to gs://{bucket_name}/{gcs_path}")
        
        logging.info(f"Successfully uploaded {source_directory} to GCS bucket {bucket_name}")
        return True
    
    except Exception as e:
        logging.error(f"Error uploading to GCS: {str(e)}")
        return False