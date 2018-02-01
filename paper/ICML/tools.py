from torch.utils.data import Dataset, DataLoader, TensorDataset
from dynnet.filters import Filter

class IndexDataset(Dataset):
    def __init__(self, dataset, indices, transform=None):
        super(IndexDataset, self).__init__()
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, ix):
        result = self.dataset[self.indices[ix]]
        if self.transform is not None:
            result = (self.transform(result[0]), result[1])
        return result

def preload_dataset(ds, batch_size):
    dummy_dl = DataLoader(ds, batch_size=10000000, shuffle=False)
    inputs, labels = next(iter(dummy_dl))
    return DataLoader(TensorDataset(*[
        x.cuda() for x in (inputs, labels)]),
        batch_size=batch_size, shuffle=False)

def compute_size(model):
    layers = []
    for module in model.modules():
        if isinstance(module, Filter):
            layers.append(int(module.get_alive_features().float().sum()))
    return layers

