# LTS-GNS
This repository contains the code for the paper [Latent task-specific Graph Network Simulators](https://arxiv.org/pdf/2311.05256.pdf) by [Philipp Dahlinger](https://github.com/philippdahlinger/), Niklas Freymuth, Michael Volpp, Tai Hoang, and Gerhard Neumann.

## Installation
Clone this repository and run

```python3 -m pip install .```

from the source directory to install all necessary public packages.
It is recommended to create a new virtual environment (``python ~= 3.10``) for this purpose.

### Internal Repositories
The following repositories are required to run the code:
- [Multi-DAFT-VI](https://github.com/ALRhub/multi_daft_vi_torch.git)
- [HMPN](https://github.com/ALRhub/hmpn.git)
- [MP_PyTorch](https://github.com/ALRhub/MP_PyTorch.git)

To install them, run
```
git clone https://www.github.com/ALRhub/multi_daft_vi_torch.git ../multi_daft_vi_torch
git clone https://www.github.com/ALRhub/HMPN.git ../hmpn
git clone https://www.github.com/ALRhub/MP_PyTorch.git ../MP_PyTorch

pip install -e ../multi_daft_vi_torch
pip install -e ../hmpn
pip install -e ../MP_PyTorch
```

### Troubleshooting
- If the following error occurs:
`ModuleNotFoundError: No module named 'torch'`
Then install torch together with torchvision and torchaudio using
`pip3 install torch torchvision torchaudio`
- If you have the problem
` fatal error: Python.h: No such file or directory`
Then install the package `python3.10-dev` using `sudo apt-get install python3.10-dev`

### Details
- Create a folder `datasets/lts_gns` in the root directory of this repository. This folder will be used
to store the datasets as `datasets/lts_gns/$dataset_name/$dataset_name_{train,test,val}.pkl`.
- Mark the src folder as source root in your IDE.
- Upgrade your torch-cluster version to GPU via
```
pip install --upgrade torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
pip install --upgrade torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
pip install --upgrade torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
```
## Run

Running experiments is done via the `main.py` script using [cw2](https://github.com/ALRhub/cw2.git).

## License
"LTS-GNS" is open-sourced under the [MIT license](LICENSE).
