
import logging
import os

class ExperimentIDFilter(logging.Filter):
    def __init__(self, exp_id):
        super().__init__()
        self.exp_id = exp_id

    def filter(self, record):
        record.exp_id = self.exp_id
        return True


def setup_logger(output_dir, exp_id, name="experiment"):
    log_file = os.path.join(output_dir, f"{exp_id}.log")

    # Set format to include experiment ID
    log_format = "%(asctime)s [%(levelname)s] [%(exp_id)s] %(message)s"

    # Create handlers
    file_handler = logging.FileHandler(log_file)
    stream_handler = logging.StreamHandler()

    # Create formatter and set it for both handlers
    formatter = logging.Formatter(log_format)
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # Add experiment ID filter to both handlers
    exp_filter = ExperimentIDFilter(exp_id)
    file_handler.addFilter(exp_filter)
    stream_handler.addFilter(exp_filter)

    # Set up the root logger
    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, stream_handler]
    )

    return logging.getLogger(name)

