import torch
import typer

from project_mlops.data import corrupt_mnist
from project_mlops.model import MyAwesomeModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def evaluate(model_checkpoint: str) -> None:
    """Evaluate a trained model (checkpoint.pth) on test data and print the accuracy.

    Args:
        model_checkpoint: Path to model checkpoint

    Returns:
        None
    """
    print("Evaluating like my life depended on it")
    print(model_checkpoint)

    model = MyAwesomeModel().to(DEVICE)  # move model to device
    model.load_state_dict(torch.load(model_checkpoint))  # load model checkpoint weights

    _, test_set = corrupt_mnist()  # load test data
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=32)  # create dataloader

    model.eval()  # set model to eval mode
    correct, total = 0, 0
    for img, target in test_dataloader:
        img, target = img.to(DEVICE), target.to(DEVICE)  # move data to device
        y_pred = model(img)  # forward pass
        correct += (y_pred.argmax(dim=1) == target).float().sum().item()
        total += target.size(0)
    print(f"Test accuracy: {correct / total}")  # print accuracy


def main() -> None:
    typer.run(evaluate)

if __name__ == "__main__":
    main()
