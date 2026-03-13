# Solar irradiance separation with deep learning

This repository contains the code and supporting files for the paper 
*“Solar irradiance separation with deep learning: An interpretable multi-task and physically constrained model based on individual-interactive features.”*

## Project Structure

### `Data/`
This directory contains the dataset used in this study:
- **`Cluster3.dat`**: Stores all available samples used in the study.

### `Model/`
This directory contains the architecture of the proposed model:
- **`model.py`**: Implements the overall framework of the model.
- **`Attention.py`**: Implements the attention mechanism used in the model.
- **`Transformer.py`**: Implements the Transformer sub-model.

### `Model_saving/`
This directory stores the trained models.

### `Result/`
This directory stores the estimated results for the testing set.

### `Utils/`
This directory contains utility functions required to run `main.ipynb`:
- **`data.py`**: Implements data loading functions.
- **`metric.py`**: Implements the calculation of evaluation metrics.
- **`train.py`**: Implements the iterative training procedure and the loss functions used for model optimization.

### `main.ipynb`
The main program of the project.

## Getting Started
1. Set up a PyTorch environment.
2. Run `main.ipynb`.

## Data and Contact
**Author:** Mengmeng Song 
**Affiliation:** School of Electrical Engineering and Automation, Harbin Institute of Technology  
**Email:** [songmeng_hit_edu@163.com](mailto:songmeng_hit_edu@163.com)  