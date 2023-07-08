import subprocess
from baselines.mlp.data import scDataset


class cTPnetDataset(scDataset):
    def __init__(self, X_path, y_path, X_transform=...):
        super().__init__(X_path, y_path, X_transform)

    def _normalize(self, data):
        pass


def savexr(data_path, output_path):
    subprocess.run(["Rscript", "./baselines/cTPnet/denoise.R", data_path, output_path])
