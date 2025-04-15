1. Install python 3.12 and poetry

2. 
`poetry install`

`poetry run pip3 install torch torchvision torchaudio`

`poetry run clearml-init` -> paste credentials to use clearml

`poetry run python fishingboatsprediction/main.py`

`nohup poetry run python fishingboatsprediction/main.py > "logs_$(date +'%Y-%m-%d_%H-%M-%S').txt" 2>&1 &`

For MacOS:
brew install proj before pyproj

TODO: 
- make clearml optional
