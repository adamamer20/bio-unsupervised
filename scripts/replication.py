from typing import Literal

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

TRAIN_SIZE = 60000


# Define a traditional single-layer neural network
class BPClassifier(nn.Module):
    def __init__(
        self, dataset_name: Literal["MNIST"], minibatch_size: int, hidden_size: int
    ):
        super().__init__()

        # Define general attributes
        self.train_data_loader, self.test_data_loader = self.load_dataset(
            dataset_name, minibatch_size
        )
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer: Adam = None

        # Define the layers
        input_size = self.train_data_loader.dataset[0][0].shape[0]
        output_size = self.train_data_loader.dataset[0][1].shape[0]
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def load_dataset(
        self, name: Literal["MNIST"], minibatch_size: int
    ) -> tuple[DataLoader, DataLoader]:
        if name == "MNIST":
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))]
            )

            train_dataset = datasets.MNIST(
                root="data", download=True, train=True, transform=transform
            )

            self.train_dataloader = DataLoader(
                train_dataset, batch_size=minibatch_size, shuffle=True
            )
            self.test_dataloader = DataLoader(
                datasets.MNIST(
                    root="data", download=True, train=False, transform=transform
                ),
                batch_size=minibatch_size,
                shuffle=True,
            )

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

    def run_epoch(
        self,
        training: bool,
    ) -> float:
        if training:
            self.train()
        else:
            self.eval()
        correct = 0
        total = 0
        data_loader = self.train_data_loader if training else self.test_data_loader
        for data, target in data_loader:
            if training and self.optimizer is not None:
                self.optimizer.zero_grad()
            outputs = self(data)
            if training:
                loss = self.criterion(outputs, target)
                loss.backward()
                self.optimizer.step()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        error_rate = 100 * (1 - correct / total)
        print(f"{'Train' if training else 'Test'} Error: {error_rate}%")
        return error_rate

    def train_and_plot_errors(
        self,
        learning_rate: float,
        epochs: int,
    ):
        self.optimizer = Adam(self.parameters(), lr=learning_rate)
        train_errors = []
        test_errors = []
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            train_error = self.run_epoch(training=True)
            train_errors.append(train_error)
            test_error = self.run_epoch(training=False)
            test_errors.append(test_error)
        self.plot_errors(train_errors, test_errors)

    def plot_errors(self, train_errors: list[float], test_errors: list[float]):
        plt.figure(figsize=(10, 5))
        plt.plot(train_errors, label="Train Error")
        plt.plot(test_errors, label="Test Error")
        plt.xlabel("Epochs")
        plt.ylabel("Error Rate (%)")
        plt.legend()
        plt.title("Training and Test Error Rates")
        plt.show()


if __name__ == "__main__":
    model = BPClassifier(input_size=784, hidden_size=2000, output_size=10)
    model.load_dataset("MNIST", 64)
    model.train_and_plot_errors(learning_rate=0.001, epochs=300)
