from clearml import Task
import torch
import numpy as np

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

set_seed(42)

def run_experiment_with_clearml():
    # Initialize a ClearML task
    task = Task.init(project_name="Your Project Name", task_name="Your Experiment Name")

    # Log hyperparameters
    params = {
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 10
    }
    task.connect(params)

    # Your training code here
    # Example: model.fit(x_train, y_train, epochs=params["epochs"], batch_size=params["batch_size"])

    # Log metrics
    task.get_logger().report_scalar("Loss", "train", value=0.1, iteration=1)
    task.get_logger().report_scalar("Accuracy", "train", value=0.95, iteration=1)


if __name__ == "__main__":
    
    run_experiment_with_clearml()