"""Pose discovery and splitting, independent of PyTorch and image decoding."""
from pathlib import Path
import re

import pandas as pd
from sklearn.model_selection import train_test_split

POSE_PATTERN = re.compile(r"P(-?\d+)_R(-?\d+)")
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg'}


def load_dataframes(root_path='./Image_Pose/robot_1_four_ball', random_state=42):
    """Return train, validation, test, pitch count and roll count.

    Image-level splits are stratified by joint pose, with 60/20/20 proportions.
    All frames carry shared P_map/R_map in DataFrame.attrs. Correlated recordings
    need a separate sequence-level evaluation protocol.
    """
    root = Path(root_path)
    if not root.is_dir():
        raise FileNotFoundError(f'Pose data directory not found: {root}')
    records = []
    for folder in sorted(root.iterdir()):
        match = POSE_PATTERN.fullmatch(folder.name)
        if not folder.is_dir() or match is None:
            continue
        pitch, roll = map(int, match.groups())
        for image in sorted(folder.iterdir()):
            if image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS:
                records.append({'image_path': str(image), 'P': pitch, 'R': roll,
                                'pose_class': f'P{pitch}_R{roll}'})
    if not records:
        raise ValueError(f'No pose images found under {root}; expected P<number>_R<number> folders')
    frame = pd.DataFrame(records)
    counts = frame.groupby('pose_class').size()
    if counts.min() < 5:
        raise ValueError('Each pose class needs at least 5 images for a stratified 60/20/20 split; '
                         f'small classes: {counts[counts < 5].to_dict()}')
    try:
        train, held_out = train_test_split(frame, test_size=0.4,
                                          stratify=frame['pose_class'], random_state=random_state)
        validation, test = train_test_split(held_out, test_size=0.5,
                                           stratify=held_out['pose_class'], random_state=random_state)
    except ValueError as exc:
        raise ValueError(f'Unable to stratify pose data; add examples per joint pose: {exc}') from exc
    p_map = {value: index for index, value in enumerate(sorted(frame['P'].unique()))}
    r_map = {value: index for index, value in enumerate(sorted(frame['R'].unique()))}
    splits = []
    for part in (train, validation, test):
        part = part.reset_index(drop=True)
        part.attrs.update(P_map=p_map.copy(), R_map=r_map.copy())
        splits.append(part)
    return (*splits, len(p_map), len(r_map))
