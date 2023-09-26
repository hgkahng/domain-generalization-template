
import typing
import numpy as np
from PIL import Image


def pil_loader(path: str) -> Image.Image:
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert("RGB")
    

def train_test_split_by_n_samples_per_class(indices: np.ndarray,
                                            labels: np.ndarray,
                                            n_samples_per_class: int = 1,
                                            random_state: int = 42,
                                            ) -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    Arguments:
        indices: np.ndarray
        labels: np.ndarray
        n_samples_per_class: int
        random_state: int
    Returns:
        a tuple of (train_indices, test_indices).
    """

    assert len(indices) == len(labels)
    unique_classes = np.unique(labels)
    
    rng = np.random.default_rng(seed=random_state)  # random number generator with seed

    train_indices = []  # buffer for training indices
    test_indices = []   # buffer for test indices

    for c in unique_classes:
        
        # get the indices where class label is `c`
        indices_with_c = indices[labels == c]

        # split `indices_with_c` into train / test
        train_indices_with_c = \
            rng.choice(indices_with_c, n_samples_per_class, replace=False)
        test_indices_with_c = \
            np.setdiff1d(indices_with_c, train_indices_with_c)
        
        # append to buffer
        train_indices.append(train_indices_with_c)
        test_indices.append(test_indices_with_c)

    # list of numpy arrays -> 1d numpy array
    train_indices = np.concatenate(train_indices, axis=0)
    test_indices = np.concatenate(test_indices, axis=0)

    return train_indices, test_indices
