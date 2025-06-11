# Iterative Optimal Transport for Multimodal Image Registration
This repository includes the implementation of our work **"Iterative Optimal Transport for Multimodal Image Registration"**.

## Repository Structure

* `example_data/`: Contains a CT–SPECT image pair with annotated landmarks.
* `demo.ipynb`: A Jupyter notebook for evaluating and visualizing the registration performance of the proposed IOT method.
* `IOT.py`: Core implementation of the Iterative Optimal Transport (IOT) algorithm.
* `utilis.py`: Utility functions for image processing and visualization.


## Environment Setup & Running the Demo

#### 1. Clone the Repository

```bash
git clone https://github.com/Mengyu8042/IOT.git
cd IOT
```

#### 2. Create and Activate a Python Environment
We recommend using [conda](https://docs.conda.io/). For example:

```bash
conda create -n iot-env python=3.8 -y
conda activate iot-env
```

#### 3. Install Required Dependencies

```bash
pip install -r requirements.txt
```

#### 4. Run the Demo

Ensure that the `example_data/` folder is present, then launch the demo:

```bash
jupyter notebook demo.ipynb
```
