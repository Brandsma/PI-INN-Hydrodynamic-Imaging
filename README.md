# Physics-Informed Invertible Neural Networks for Hydrodynamic Imaging

This repository contains the code for a master's thesis on using Physics-Informed Invertible Neural Networks (PI-INNs) for hydrodynamic imaging. The goal of the project is to determine the position and properties of an object in a fluid by observing the fluid's velocity field, which is a challenging inverse problem.

This codebase has been refactored and polished from the original research code to showcase a high standard of code quality, documentation, and reproducibility for professional and academic review.

## Project Overview

The core of this project is the application of Invertible Neural Networks (INNs) to solve the inverse problem of hydrodynamic imaging. By formulating the forward process (object properties to velocity field) as an invertible transformation, the INN can be trained to directly learn the inverse mapping (velocity field to object properties).

Furthermore, the model is enhanced by incorporating physical laws as a soft constraint in the loss function, creating a Physics-Informed Neural Network (PINN). This encourages the model to learn solutions that are consistent with the underlying physics of fluid dynamics, improving accuracy and generalizability.

The repository includes implementations for:
-   **LSTM Model:** A baseline sequence-to-sequence model.
-   **Invertible Neural Network (INN):** The core model for solving the inverse problem.
-   **Physics-Informed INN (PINN):** The INN enhanced with a physics-based loss function.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/master-thesis-hydrodynamic-imaging.git
    cd master-thesis-hydrodynamic-imaging
    ```

2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The main scripts for training the models are located in the `src` directory.

### Training the INN/PINN Model

The `src/INN/main.py` script is used to train the Invertible Neural Network.

**To train the standard INN:**
```bash
python src/INN/main.py
```

**To train the Physics-Informed INN (PINN):**
```bash
python src/INN/main.py --use_pde
```

### Training the LSTM Model

The `src/LSTM/train_lstm.py` script is used to train the baseline LSTM model. You can specify the training data location and the output directory for the trained model.

```bash
python src/LSTM/train_lstm.py --train_loc /path/to/your/data.npy --model_dir /path/to/save/model/
```

## Repository Structure

```
.
├── src/
│   ├── INN/            # Source code for the INN and PINN models
│   │   ├── main.py     # Main script to train the INN/PINN
│   │   ├── inn.py      # INN model definition
│   │   ├── hydro.py    # Hydrodynamic data loading and PDE loss
│   │   ├── trainer.py  # Custom Keras trainer
│   │   └── ...
│   ├── LSTM/           # Source code for the LSTM model
│   │   ├── train_lstm.py # Main script to train the LSTM
│   │   └── ...
│   ├── lib/            # Shared library code
│   └── ...
├── data/               # (Not included) Placeholder for simulation data
├── trained_models/     # (Not included) Placeholder for trained models
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Scientific Context

For a detailed description of the research, including the mathematical formulations and the physics behind the model, please refer to the project description at:
[https://abebrandsma.com/project/master-thesis-hydrodynamic-imaging-physics](https://abebrandsma.com/project/master-thesis-hydrodynamic-imaging-physics)
*(Note: This link was not accessible at the time of this code review, so the implementation of the PDE was based solely on the source code.)*
