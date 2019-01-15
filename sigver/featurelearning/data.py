import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torchvision import transforms as transforms


class TransformDataset(Dataset):
    """
        Dataset that applies a transform on the data points on __get__item.
    """
    def __init__(self, dataset, transform, transform_index=0):
        self.dataset = dataset
        self.transform = transform
        self.transform_index = transform_index

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = self.dataset[item]
        img = data[self.transform_index]

        return tuple((self.transform(img), *data[1:]))


def extract_features(x, process_function, batch_size, input_size=None):
    data = TensorDataset(torch.from_numpy(x))

    if input_size is not None:
        data_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
        ])

        data = TransformDataset(data, data_transforms)

    data_loader = DataLoader(data, batch_size=batch_size)
    result = []

    with torch.no_grad():
        for batch in data_loader:
            result.append(process_function(batch))
    return torch.cat(result).cpu().numpy()
