# `veloce`


<p align="center">
  <img src="https://github.com/user-attachments/assets/db1ebeb2-79db-4d2a-990a-6d6227206a84" width="52%"
 alt="veloce_logo"/>
</p>
<div align="center">
  
![](https://img.shields.io/badge/Python-181717?style=plastic&logo=python)
![](https://img.shields.io/badge/Author-Davide%20Piras%20-181717?style=plastic)
![](https://img.shields.io/badge/Installation-pip%20install%20coming_soon-181717?style=plastic)
[![arXiv](https://img.shields.io/badge/arXiv-2504.10453-b31b1b.svg)](https://arxiv.org/abs//2504.10453)

</div>


Welcome to ``veloce``, the velocity power spectrum covariance emulator! This repository and the corresponding package will be made publicly available upon publication of the corresponding paper. If you are interested in early access please [raise an issue](https://github.com/dpiras/veloce/issues) or contact [Davide Piras](mailto:dr.davide.piras@gmail.com). 



## Installation

To use the emulator and/or sample your supernovae posterior, follow these steps:
1. (optional) `conda create -n veloce python=3.11 jupyter` (create a custom `conda` environment with python 3.11) 
2. (optional) `conda activate vveloce` (activate it)
3. Install the package:

        pip install velocemu
        python -c 'import velocemu'

   or alternatively, clone the repository and install it:

        git clone https://github.com/dpiras/veloce.git
        cd veloce
        pip install . 
        python -c 'import velocemu'

## Usage

The latter option will also give you access to all [Jupyter notebooks](https://github.com/dpiras/VAExEDE/blob/main/notebooks/quickstart.ipynb), which include information on how to generate a single element of the covariance, use the emulator, and sample the posterior.

## Trained models

You can find the available models in this folder (TBC), which will be updated when new models become available. Currently, we provide the models that lead to the final results of the paper, namely the nonlinear case with fixed $\sigma_{\rm u}$, but more models are in production. If you are interested in other models, please [reach out](https://github.com/dpiras/veloce/issues) or contact [Davide Piras](mailto:dr.davide.piras@gmail.com). Also note that it should be straightforward for you to train your own models using [CosmoPower](https://github.com/alessiospuriomancini/cosmopower).

## Contributing and contacts

Feel free to [fork](https://github.com/dpiras/veloce/fork) this repository to work on it; otherwise, please [raise an issue](https://github.com/dpiras/veloce/issues) or contact [Davide Piras](mailto:dr.davide.piras@gmail.com).

## Citation

If you use `veloce`, please cite the corresponding paper:

     @ARTICLE{Piras25,
          author = {{Piras}, Davide and {Sorrenti}, Francesco and {Durrer}, Ruth and {Kunz}, Martin},
          title = "{Anchors no more: Using peculiar velocities to constrain $H_0$ and the primordial Universe without calibrators}",
          journal = {arXiv e-prints},
          year = 2025,
          month = apr,
          eid = {arXiv:2504.10453},
          pages = {arXiv:2504.10453},
          doi = {10.48550/arXiv.2504.10453},
          archivePrefix = {arXiv},
          eprint = {2504.10453},
          primaryClass = {astro-ph.CO},
          adsurl = {https://ui.adsabs.harvard.edu/abs/2025arXiv250410453P},
     }



## License

`veloce` is released under the GPL-3 license (see [LICENSE](https://github.com/dpiras/veloce/blob/main/LICENSE.txt)) subject to the non-commercial use condition.

     veloce
     Copyright (C) 2025 Davide Piras & contributors

     This program is released under the GPL-3 license (see LICENSE.txt), subject to a non-commercial use condition.
