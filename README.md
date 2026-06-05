# MLOps project by Anne, Lærke and Stoyan
This repository contains the code that form our MLOps implementations.

# The code consist of:


## The code is structured into multiple folders:
- **conf:** contains hydra-configurations files.
- **data:** contains all code used for the test of the different architectures and loading of the different datasets.
- **experiments:** contains all code used in generating the synthetic dataset, together with the code for the initial evaluation of the dataset: model architecture, training and validation script.
- **model:** contains all code used for the test of the different architectures and loading of the different datasets.
- **monitoring:** contains all code used in generating the synthetic dataset, together with the code for the initial evaluation of the dataset: model architecture, training and validation script.
- **outputs:** contains all code used for the test of the different architectures and loading of the different datasets.
- **unitests:** contains all code used in generating the synthetic dataset, together with the code for the initial evaluation of the dataset: model architecture, training and validation script.


# To create virtual environment for development do:
1. Create .venv folder by writing in terminal
    $ python3.11.9 -m venv .venv
2. Activate virtual enviroment:
    $ source .venv/bin/activate
3. Install requirements:
    $ python -m pip install -r requirements-dev.txt
4. All good to go!
