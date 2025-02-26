import yaml
from addict import Dict
from datetime import datetime
import os
import shutil  # Add shutil for file copying

def load_config(path):
    with open(path) as file:
        config_dict = yaml.safe_load(file)
    return Dict(config_dict)

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
    
    # Copy data.json from data directory to output directory and rename it to data_config.json
    if config.DATASET == 'os':
        data_source = os.path.join("os", "data", "config.json")
        data_destination = os.path.join(output_dir, "data_config.json")
        if os.path.exists(data_source):
            shutil.copy2(data_source, data_destination)
    
    return output_dir