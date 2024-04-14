import yaml

def deep_merge(dct, merge_dct):
    """ Recursive dict merge. Inspired by dict.update(), instead merges dicts and sets."""
    for k, v in merge_dct.items():
        if (k in dct and isinstance(dct[k], dict)
                and isinstance(merge_dct[k], dict)):
            deep_merge(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]

def load_and_merge_yaml(base_filepath, override_filepath):
    """Load and merge two yaml files. The override file takes precedence over the base file."""
    with open(base_filepath, 'r') as file:
        base_config = yaml.safe_load(file)
    with open(override_filepath, 'r') as file:
        override_config = yaml.safe_load(file)
    
    deep_merge(base_config, override_config)
    return base_config

if __name__ == "__main__": 
    # Example usage
    base_filepath = 'path/to/base_config.yaml'
    override_filepath = 'path/to/child_config.yaml'
    merged_config = load_and_merge_yaml(base_filepath, override_filepath)
    print(merged_config)
