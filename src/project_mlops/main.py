import matplotlib.pyplot as plt
import torch
import typer
from data import corrupt_mnist
from project_mlops.model import MyAwesomeModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

app = typer.Typer()


@app.command()
def train(lr: float = 1e-3, batch_size: int = 32, epochs: int = 10) -> None:
    """Train a model on MNIST.

    args:
        lr: Learning rate
        batch_size: Batch size
        epochs: Number of epochs

    returns:
        None
    """
    print("Training day and night")
    print(f"{lr=}, {batch_size=}, {epochs=}")

    # initialize model and load data
    model = MyAwesomeModel().to(DEVICE)
    train_set, _ = corrupt_mnist()

    # create dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

    # define loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    statistics = {"train_loss": [], "train_accuracy": []}  # store stats
    for epoch in range(epochs):
        model.train()  # set model to train mode
        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)  # move data to device
            optimizer.zero_grad()  # zero gradients
            y_pred = model(img)  # forward pass
            loss = loss_fn(y_pred, target)  # compute loss
            loss.backward()  # backward pass
            optimizer.step()  # update weights
            statistics["train_loss"].append(loss.item())  # save loss

            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()  # compute accuracy
            statistics["train_accuracy"].append(accuracy)  # save accuracy

            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}, accuracy: {accuracy}")

    print("Training complete")
    torch.save(model.state_dict(), "model.pth")  # save model -> models/model.pth
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")
    fig.savefig("training_statistics.png")  # save figure -> reports/figures/training_statistics.png


if __name__ == "__main__":
    typer.run(train)
