import torch.multiprocessing as mp

mp.set_start_method(method="spawn", force=True)
