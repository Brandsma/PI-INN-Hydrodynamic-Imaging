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

## Requirements

- **Python Version:** This project has been tested on Python 3.11.9
- **Operating System:** Developed and tested on macOS
- **Dependencies:** Listed in `requirements.txt`

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/master-thesis-hydrodynamic-imaging.git
    cd master-thesis-hydrodynamic-imaging
    ```

2.  Create and activate a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Quick Start

After setting up the virtual environment, the easiest way to run the project is using the provided example script:

```bash
python run_example.py
```

This script will:
- Run a minimal INN example on synthetic sine wave data
- Run a Physics-Informed INN (PINN) example with PDE constraints
- Use a small number of epochs for quick demonstration
- Save trained models to `./trained_models_example/`

## Usage

### For Advanced Users

The main scripts for training the models are located in the `src` directory and should be run as Python modules from the project root.

### Training the INN/PINN Model

**To train the standard INN:**
```bash
python -m src.INN.main
```

**To train the Physics-Informed INN (PINN):**
```bash
# Edit src/INN/main.py to set use_pde=True in the simple_run() call
python -m src.INN.main
```

### Training the LSTM Model

The `src/LSTM/train_lstm.py` script is used to train the baseline LSTM model. You can specify the training data location and the output directory for the trained model.

```bash
python -m src.LSTM.train_lstm --train_loc /path/to/your/data.npy --model_dir /path/to/save/model/
```

**Note:** The advanced usage requires proper simulation data files. These are not provided. For a quick demonstration without data dependencies, use the `run_example.py` script instead.

## Troubleshooting

### Import Errors
If you encounter `ImportError` messages, make sure to:
1. Activate your virtual environment: `source venv/bin/activate`
2. Run scripts from the project root directory
3. Use the module execution format: `python -m src.INN.main` instead of `python src/INN/main.py`

### Missing Data Files
The `run_example.py` script uses synthetic data and should work without additional data files. For full functionality with real simulation data, you'll need to provide the appropriate `.npy` data files in the expected locations.

### Python Version Compatibility
This project was developed on Python 3.7. This project was later tested again on Python 3.11.9. While it may work with other Python 3.x versions, I recommend using Python 3.11 for best compatibility.

## Repository Structure

```
.
├── src/
│   ├── __init__.py     # Package initialization
│   ├── INN/            # Source code for the INN and PINN models
│   │   ├── __init__.py # Package initialization
│   │   ├── main.py     # Main script to train the INN/PINN
│   │   ├── inn.py      # INN model definition
│   │   ├── hydro.py    # Hydrodynamic data loading and PDE loss
│   │   ├── trainer.py  # Custom Keras trainer
│   │   ├── data.py     # Data loading utilities
│   │   ├── flow.py     # Flow model implementations
│   │   ├── utils.py    # Utility functions
│   │   └── sine.py     # Sine wave data generation
│   ├── LSTM/           # Source code for the LSTM model
│   │   ├── train_lstm.py # Main script to train the LSTM
│   │   └── ...
│   └── lib/            # Shared library code
│       ├── __init__.py # Package initialization
│       └── params.py   # Parameter definitions
├── run_example.py      # Quick start example script
├── requirements.txt    # Python dependencies
├── RUNNING_INN.md      # Detailed running instructions
├── venv/               # Virtual environment (created after setup)
├── data/               # (Not included) Placeholder for simulation data
├── trained_models/     # (Not included) Placeholder for trained models
└── README.md           # This file
```

## Data Source and Sensor Design

This research is based on hydrodynamic imaging using fiber optic sensors that detect fluid deflections caused by moving objects. The sensor system employs Fibre Bragg Gratings (FBGs) placed at various locations to capture deflection data as objects move through the fluid environment. For a more detailed explanation of the sensor design and data generation, please refer to the [project description and original thesis paper](https://abebrandsma.com/project/master-thesis-hydrodynamic-imaging-physics).

### Sensor Design

![Sensor Design Schematic](Thesis/images/sensor_design_schematic.pdf)

The sensor array uses Fibre Bragg Gratings to measure fluid deflections at multiple spatial positions. These sensors provide high-precision measurements of the hydrodynamic disturbances created by objects moving through the fluid.

### Wavelet Response Profiles

![Wavelet Profile Family](Thesis/images/wavelet_profile_family.pdf)

The wavelet profile family demonstrates how the sensors respond to a sphere moving past the sensor array at different positions. Each profile shows the characteristic signature that different object trajectories create in the sensor readings, which forms the basis for the inverse problem solved by the neural networks in this project.

## Scientific Context

For a detailed description of the research, including the mathematical formulations and the physics behind the model, please refer to the project description at:
[https://abebrandsma.com/project/master-thesis-hydrodynamic-imaging-physics](https://abebrandsma.com/project/master-thesis-hydrodynamic-imaging-physics)

This link contains a comprehensive blog post and reference to the original thesis paper with proper attribution for all images and research content.
