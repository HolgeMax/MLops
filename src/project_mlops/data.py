import glob
import os

import torch
import typer


def normalize(images: torch.Tensor) -> torch.Tensor:
    """Normalize images to z-score standardization.

    Args:
        images: Tensor (N, C, H, W)

    Returns:
        normalized images: Tensor (N, C, H, W)
    """
    return (images - images.mean()) / images.std()


def preprocess_data(raw_dir: str, processed_dir: str) -> None:
    """Process raw data, both train and test, and save it to processed directory.

    Args:
        raw_dir: Raw data directory
        processed_dir: Processed data save directory
    
    Returns:
        None
    """
    # Load and concatenate train data
    train_images, train_target = [], [] # create empty lists
    images_paths = sorted(glob.glob(os.path.join(raw_dir, "train_images_*.pt"))) 
    targets_paths = sorted(glob.glob(os.path.join(raw_dir, "train_target_*.pt")))
    if not images_paths: # check if list is empty
        raise ValueError(f"No train_images files found in {raw_dir}") # if empty -> raise error
    if len(images_paths) != len(targets_paths): # check input/target len match
        raise ValueError(
            f"Mismatched number of train images ({len(images_paths)}) and targets ({len(targets_paths)}) in {raw_dir}"
        )
    # append onto lists
    for img_p, tgt_p in zip(images_paths, targets_paths):
        train_images.append(torch.load(img_p))
        train_target.append(torch.load(tgt_p)) 
    
    # create single dir tensors
    train_images = torch.cat(train_images)
    train_target = torch.cat(train_target)

    # Load test data
    test_images: torch.Tensor = torch.load(f"{raw_dir}/test_images.pt")
    test_target: torch.Tensor = torch.load(f"{raw_dir}/test_target.pt")


    train_images = train_images.unsqueeze(1).float() # use unsqueeze to add channel dim
    test_images = test_images.unsqueeze(1).float()
    train_target = train_target.long() # ensure target is long dtype for CE loss
    test_target = test_target.long()

    # use def normalize
    train_images = normalize(train_images)
    test_images = normalize(test_images)

    # save processed data
    torch.save(train_images, f"{processed_dir}/train_images.pt")
    torch.save(train_target, f"{processed_dir}/train_target.pt")
    torch.save(test_images, f"{processed_dir}/test_images.pt")
    torch.save(test_target, f"{processed_dir}/test_target.pt")


def corrupt_mnist() -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Return train and test datasets for corrupt MNIST.
    
    Args:
        None
    
    Returns:
        train_set: torch.utils.data.Dataset
    """
    # load processed data
    train_images = torch.load("data/processed/train_images.pt")
    train_target = torch.load("data/processed/train_target.pt")
    test_images = torch.load("data/processed/test_images.pt")
    test_target = torch.load("data/processed/test_target.pt")

    train_set = torch.utils.data.TensorDataset(train_images, train_target)
    test_set = torch.utils.data.TensorDataset(test_images, test_target)
    return train_set, test_set 


def main() -> None:
    typer.run(preprocess_data)

if __name__ == "__main__":
    main()