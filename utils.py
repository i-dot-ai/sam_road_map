import yaml
from addict import Dict
from datetime import datetime
import os
import shutil  # Add shutil for file copying

def load_config(path):
    with open(path) as file:
        config_dict = yaml.safe_load(file)
        
    if config_dict['DATASET'] == 'os':
        data_config_path = os.path.join("os", "data", "config.json")
        with open(data_config_path) as file:
            data_config_dict = yaml.safe_load(file)
        config_dict['DATA_CONFIG'] = data_config_dict
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
    
    return output_dir