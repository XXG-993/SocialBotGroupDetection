import yaml


class ReadConfig:
    def __init__(self):
        # TODO: change singleton style or other avoid hard encoding
        self.cfg_path = "LSTM-AE/config.yaml"

    def read_config(self) -> dict:
        with open(self.cfg_path, 'r')as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        return data
