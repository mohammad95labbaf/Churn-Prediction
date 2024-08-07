# config.py

import yaml

class Config:
    def __init__(self, config_file):
        self.config_file = config_file
        self.config = self.load_config()

    def load_config(self):
        with open(self.config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def get_data_settings(self):
        return self.config['data']

    def get_model_settings(self):
        return self.config['model']

    def get_classifier_settings(self):
        return self.config['classifiers']