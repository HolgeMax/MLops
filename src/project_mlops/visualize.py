import matplotlib.pyplot as plt
import torch
import typer
from project_mlops.model import MyAwesomeModel
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def visualize(model_checkpoint: str, figure_name: str = "embeddings.png") -> None:
    """Visualize model predictions, and save as a figure.

    args:
        model_checkpoint: Path to model checkpoint
        figure_name: Name of the output figure file

    returns:
        None
    """
    model: torch.nn.Module = MyAwesomeModel().to(DEVICE)

    # load weights onto correct device
    state_dict = torch.load(model_checkpoint, map_location=DEVICE)
    model.load_state_dict(state_dict)

    model.eval()
    model.fc1 = torch.nn.Identity()

    test_images = torch.load("data/processed/test_images.pt")
    test_target = torch.load("data/processed/test_target.pt")
    test_dataset = torch.utils.data.TensorDataset(test_images, test_target)

    embeddings, targets = [], []

    loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

    with torch.inference_mode():
        for images, target in loader:
            # move data to device
            images = images.to(DEVICE)
            target = target.to(DEVICE)

            preds = model(images)
            embeddings.append(preds)
            targets.append(target)

    # move back to CPU before numpy
    embeddings = torch.cat(embeddings).cpu().numpy()
    targets = torch.cat(targets).cpu().numpy()

    if embeddings.shape[1] > 500:
        pca = PCA(n_components=100)
        embeddings = pca.fit_transform(embeddings)

    tsne = TSNE(n_components=2)
    embeddings = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 10))
    for i in range(10):
        mask = targets == i
        plt.scatter(embeddings[mask, 0], embeddings[mask, 1], label=str(i))

    plt.legend()
    plt.savefig(f"reports/figures/{figure_name}")


if __name__ == "__main__":
    typer.run(visualize)
