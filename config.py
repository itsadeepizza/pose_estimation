from pathlib import Path

class Config():
    # PARAMETERS HAD TO BE UPPERCASE !

    root = Path(__file__).resolve().parents[1]

    WINDOW_NAME = 'MediaPipe Hands'
    VIDEO_SOURCE = 0

    RESOLUTION = (640, 480)

    LEN_QUEUE_MEDIAPIPE = 10



    def __init__(self):

        self.set_derivate_parameters()

    def set_derivate_parameters(config):
        """Set parameters which are derivate from other parameters"""
        config.PATH_DATASET = str(config.root / 'dataset')
        config.PATH_CSV     = str(config.root / 'dataset/feelit/feelit.tsv')
        config.ROOT_RUNS    = str(config.root)


    def get(self, key, default_return_value=None):
        """Safe metod to get an attribute. If the attribute does not exist it returns
        None or a specified default value"""
        if hasattr(self, key):
            return self.__getattribute__(key)
        else:
            return default_return_value

selected_config = Config()

if __name__ == '__main__':
    pass
