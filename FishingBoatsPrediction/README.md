`poetry install`
`pip3 install torch torchvision torchaudio`

<!-- https://api.vtexplorer.com/docs/ref-aistypes.html -->

Questions:

- should we predict for all vessels or only fishing boats? If only fishing boats - how to destinguish fishing boats in the training dataset?
- what should be th input and output?

1. Past trajectory => Future trajectory
2. Past+Future trajectories => Trajectory in the middle

- own length of the context window, missng length 100. from 0 to 1000
