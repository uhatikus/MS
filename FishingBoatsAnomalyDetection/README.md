# Fishing Boats Prediction

This project aims to predict the trajectory of fishing boats using machine learning techniques, specifically Stable Diffusion, due to the random nature of fishing boat movements. It leverages AIS (Automatic Identification System) datasets to train and evaluate the model.

## Dataset

The dataset comprises AIS data, which includes historical trajectories of fishing boats. An example of AIS data can be found [here](https://api.vtexplorer.com/docs/response-ais.html).

## Model

The project uses Stable Diffusion, an AI-based generative model, to predict fishing boat trajectories, chosen for its ability to handle the stochastic nature of boat movements.

## Usage

To set up and run the project, follow these steps:

1. **Install Python 3.12 and Poetry**:

   - Ensure you have Python 3.12 installed.
   - Install [Poetry](https://python-poetry.org/), a dependency management tool.

2. **Install Dependencies and Run**:
   ```bash
   poetry install
   ```
   ```bash
   poetry run pip3 install torch torchvision torchaudio
   ```
   - if you would like to use clearml run the following command and paste your ClearML credentials to enable experiment tracking:
   ```bash
   poetry run clearml-init
   ```
   ```bash
   poetry run python fishingboatsprediction/main.py
   ```
   - For background execution with logging:
     ```bash
     nohup poetry run python fishingboatsprediction/main.py > "logs_$(date +'%Y-%m-%d_%H-%M-%S').txt" 2>&1 &
     ```

**Note for macOS Users**:

- Before installing `pyproj`, ensure you have `proj` installed via Homebrew:
  ```bash
  brew install proj
  ```

## License

This project is licensed under the MIT License.
