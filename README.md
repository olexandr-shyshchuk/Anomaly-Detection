# Project Name

## Overview

This project focuses on the development of a data processing and anomaly detection pipeline for financial data, specifically Bitcoin prices. The project consists of three Jupyter notebooks and a Python script:

1. **emmisions_finding_develop.ipynb**: This notebook explores the development of a class named `NextStepWinsorize` in the `next_step.py` script. The class is designed to perform data preprocessing by applying the Winsorizing technique and identifying outliers.

2. **model_develop.ipynb**: This notebook continues the development of the `NextStepWinsorize` class, focusing on the generation of synthetic data with anomaly emulation. It also explores the fitting of probability distributions to the generated data for anomaly detection.

3. **main.ipynb**: The main notebook demonstrates the complete workflow by utilizing the functionalities developed in the previous notebooks. It showcases anomaly detection, data extrapolation, and other methods for identifying anomalies.

4. **next_step.py**: This Python script contains the implementation of the `NextStepWinsorize` class, which encapsulates the Winsorizing process and anomaly emulation functionalities.

## Files

- **emmisions_finding_develop.ipynb**
- **model_develop.ipynb**
- **main.ipynb**
- **next_step.py**

## Usage

To run the project:

1. Execute the code in the notebooks in the specified order: `emmisions_finding_develop.ipynb`, `model_develop.ipynb`, and `main.ipynb`.
2. Ensure that the `price.csv` file containing Bitcoin price data is available in the project directory.

## Requirements

The project relies on the following dependencies:

- Python 3.x
- NumPy
- pandas
- matplotlib
- SciPy

Install these dependencies using the following command:

```bash
pip install numpy pandas matplotlib scipy
```

## `next_step.py`

### NextStepWinsorize Class

The `NextStepWinsorize` class encapsulates the Winsorizing process and anomaly emulation functionalities. The class provides the following methods:

- **`fit_transform(data, N=13)`**: Fits the Winsorizing transformation to the input data and transforms it. The parameter `N` controls the grouping of data points.
  
- **`show_winsorized()`**: Visualizes the original data and the Winsorized data.
  
- **`show_outliners()`**: Visualizes the data with highlighted emission points (anomalies).

- **`generate_data(N=80, emmisions=True, prob=0.15)`**: Generates synthetic data with anomaly emulation based on the fitted distributions.

- **`show_generated_data()`**: Visualizes the generated synthetic data.

## Example Usage

```python
import numpy as np
import pandas as pd
from next_step import NextStepWinsorize

# Load Bitcoin price data
data = np.array(pd.read_csv('price.csv')['Bitcoin'])

# Initialize NextStepWinsorize class
wins = NextStepWinsorize()

# Fit and transform data
wins.fit_transform(data)

# Visualize Winsorized data
wins.show_winsorized()

# Visualize data with highlighted emission points
wins.show_outliners()

# Generate synthetic data
generated_data = wins.generate_data()

# Visualize the generated synthetic data
wins.show_generated_data()
```

**Note:** Ensure that the `price.csv` file is present in the project directory.
