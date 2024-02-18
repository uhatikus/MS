from torch.utils.data import DataLoader

from src import dataset
import yaml


def train_model(config):
    # read values from config
    train_data_filename = config.get("train_data_filename")
    valid_data_filename = config.get("valid_data_filename")
    max_seqlen = config.get("max_seqlen")
    device = config.get("device", "cpu")
    batch_size = config.get("batch_size", 32)

    # prepare data
    train_data = dataset.AISDataset(train_data_filename, max_seqlen, device)
    valid_data = dataset.AISDataset(valid_data_filename, max_seqlen, device)

    # creaete data loader
    train_data_loader = DataLoader(train_data, batch_size, shuffle=True)
    valid_data_loader = DataLoader(valid_data, batch_size, shuffle=True)

    # model
    model = models.TrAISformer(cf, partition_model=None)


def test_model(config):
    # read values from config
    test_data_filename = config.get("test_data_filename")
    max_seqlen = config.get("max_seqlen")
    device = config.get("device", "cpu")

    # prepare data
    test_data = dataset.AISDataset(test_data_filename, max_seqlen, device)


if __name__ == "__main__":
    # read config
    with open("src/config.yaml", "r") as yaml_file:
        config = yaml.safe_load(yaml_file)
    # train
    train_model(config)
