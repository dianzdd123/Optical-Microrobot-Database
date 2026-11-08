"""PyTorch image wrapper for pose data; see pose_data for splitting rules."""
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# Re-export the loader for the existing experiment scripts.
from pose_data import load_dataframes


class PoseDataset(Dataset):
    """Return image, pitch class and roll class using shared label mappings.

    Mappings default to those attached by load_dataframes, then to this frame's
    labels. Pass explicit mappings when using independently constructed frames.
    """
    def __init__(self, dataframe, transform=None, p_map=None, r_map=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform if transform is not None else transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor()])
        if p_map is None:
            p_map = dataframe.attrs.get('P_map')
        if r_map is None:
            r_map = dataframe.attrs.get('R_map')
        self.P_map = dict(p_map) if p_map is not None else {
            value: index for index, value in enumerate(sorted(self.df['P'].unique()))}
        self.R_map = dict(r_map) if r_map is not None else {
            value: index for index, value in enumerate(sorted(self.df['R'].unique()))}
        for column, mapping in (('P', self.P_map), ('R', self.R_map)):
            unknown = set(self.df[column]) - set(mapping)
            if unknown:
                raise ValueError(f'Unmapped {column} labels: {sorted(unknown)}')

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        with Image.open(row['image_path']) as source:
            image = source.convert('RGB')
        return self.transform(image), self.P_map[row['P']], self.R_map[row['R']]

    def __len__(self):
        return len(self.df)
