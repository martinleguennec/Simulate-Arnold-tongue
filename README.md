# Simulate Arnold tongue

This repository contains code to simulated a forced Haken-Kelso-Bunz (HKB) oscillator under varying coupling strengths and forcing frequencies. The goal is to visualize the regions of synchronization—referred to as **Arnold tongues**—by analyzing the dispersion of the relative phase (SD $\phi$).

Low values of SD $\phi$ indicate stable synchronization, allowing the identification of the parameter space (frequency × coupling) that defines the Arnold tongue.

## Code execution

To run the simulation:

1. Open the Jupyter notebook main.ipynb.

2. Execute each cell in sequence to simulate the oscillator dynamics and generate the Arnold tongue visualization.

> For guidance on running Jupyter notebooks in Visual Studio Code, refer to the official documentation:
> https://code.visualstudio.com/docs/datascience/jupyter-notebooks

### Virtual Environment (recommended)

To ensure reproducibility and avoid dependency conflicts, it is recommended to use a virtual environment.

1. Open a terminal in the repository directory.

2. Create a virtual environment:

    On macOS and Linux:

    ```bash
    python3 -m venv .venv
    ```

    On Windows:

    ```cmd
    python -m venv .venv
    ```

3. Activate the virtual environment:

    On macOS and Linux:

    ```bash
    source .venv/bin/activate
    ```

    On Windows (Command Prompt):

    ```cmd
    .venv\Scripts\activate
    ```

4. Install the required packages:

    ```bash
    pip install numpy matplotlib scipy
    ```
