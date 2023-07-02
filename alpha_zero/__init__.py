import logging

import torch.multiprocessing as mp
from rich.logging import RichHandler

logging.basicConfig(
    format="%(message)s",
    datefmt="[%Y-%m-%dT%X]",
    level=logging.INFO,
    handlers=[RichHandler()],
)

mp.set_start_method(method="spawn", force=True)
