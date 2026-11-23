# Deployment of Large IQP Circuit Born Machines on Qubit-Limited Hardware Using Noise Injection

A repository for the code and data used for the paper of the same name, currently available as a preprint: https://arxiv.org/abs/2609.35163

## Structure

```bash
├── /dist_iqp
│   ├── analysis.py                 # Analysis functions
│   ├── ibm_experiment.py           # Circuit distributor for IBM hardware experiments
│   ├── noiseless_node.py           # Circuit distributor for noiseless simulation
│   └── utils.py                    # Various utility functions
├── /shallow_28q
│   ├── /extensive_run_results      # Sampled bitstrings
│   ├── /trained_params_iqp         # Trained models from https://github.com/uriballo/IQP-NISQ
│   ├── extensive_runs.py           # Sampling runs
│   ├── ~correlator_comparison~.py  # Analysis of samples - Expectation value accuracy
│   └── draw~.ipynb                 # Jupyter notebooks for drawing paper figures
├── /ibm
│   ├── /ibm_results                # Experimental settings and results
│   ├── /noise_fitting_results      # Hardware noise modeling settings and results
│   ├── large_iqp_percolation.ipynb # Noise fitting, deployment, and error mitigation
│   ├── ibm_correlator_test.py      # Analysis of samples - Expectation value accuracy
│   ├── mmd_analysis.ipynb          # Analysis of samples - MMD calculation
│   └── draw~.ipynb                 # Jupyter notebooks for drawing paper figures
├── requirements.txt
└── README.md
``` 

## Dependencies

- This uses `iqpopt` version 2024.7.0. Install this through the instructions in https://github.com/XanaduAI/iqpopt
- Otherwise, install the packages in `requirements.txt`
- Paper results were created using Python 3.12.12