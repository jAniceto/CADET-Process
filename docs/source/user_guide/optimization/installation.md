# installation of different optimizers

## Ax/BoTorch

for local use, first install pytorch as CPU only version if you don't have
a GPU to run on (see https://pytorch.org/get-started/locally/).

```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

then install ax and botorch

```bash
pip3 install ax-platform
```

this will install a lot of dependencies via pip which makes maintenance of the conda environment increasingly difficult. Therefore, in the future it
would be preferable to set up a list of requirements and install as many of them via `conda`
