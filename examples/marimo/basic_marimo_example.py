import marimo

__generated_with = "0.19.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md("# MNIST Neural Network Training")
    return (mo,)


@app.cell
def _():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    return DataLoader, datasets, nn, optim, torch, transforms


@app.cell
def _(DataLoader, datasets, transforms):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST("./data", train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000)

    return test_loader, train_loader


@app.cell
def _(nn):
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(28 * 28, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 10)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = x.view(-1, 28 * 28)
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            return self.fc3(x)

    return (Net,)


@app.cell
def _(Net, mo, nn, optim, test_loader, torch, train_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    epochs = 5
    history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (output.argmax(1) == target).sum().item()
            total += len(target)

        train_acc = correct / total

        # Evaluate
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_correct += (output.argmax(1) == target).sum().item()
                test_total += len(target)

        test_acc = test_correct / test_total
        history.append({"epoch": epoch + 1, "loss": total_loss / len(train_loader), "train_acc": train_acc, "test_acc": test_acc})

    mo.md(
        "\n".join(
            [f"**Epoch {h['epoch']}** — Loss: {h['loss']:.4f}, Train Acc: {h['train_acc']:.4f}, Test Acc: {h['test_acc']:.4f}" for h in history]
        )
    )

    return (history,)


@app.cell
def _(history, mo):
    import altair as alt
    import pandas as pd

    df = pd.DataFrame(history)
    chart = alt.Chart(df).transform_fold(
        ["train_acc", "test_acc"], as_=["metric", "accuracy"]
    ).mark_line(point=True).encode(
        x="epoch:O",
        y=alt.Y("accuracy:Q", scale=alt.Scale(domain=[0.9, 1.0])),
        color="metric:N",
    ).properties(width=500, title="Accuracy over Epochs")

    mo.ui.altair_chart(chart)
    return


if __name__ == "__main__":
    app.run()
