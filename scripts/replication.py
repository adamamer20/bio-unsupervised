from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.integrate import solve_ivp
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Classifier(nn.Module):
    train_data_loader: DataLoader
    test_data_loader: DataLoader
    criterion: nn.CrossEntropyLoss
    optimizer: Adam

    input_size: int
    hidden_size: int
    output_size: int

    def __init__(
        self, dataset_name: Literal["MNIST"], minibatch_size: int, hidden_size: int
    ):
        super().__init__()
        self.train_data_loader, self.test_data_loader = self._load_dataset(
            dataset_name, minibatch_size
        )

        # Initialize hidden layer weights W (hidden_size x input_size)
        input_size = self.train_data_loader.dataset[0][0].shape[0]
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = len(self.test_data_loader.dataset.classes)

        # Initialize layers
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        # Initialize training
        self.train_errors = []
        self.test_errors = []
        self.criterion = nn.CrossEntropyLoss()

        self.to(device)

    def forward(x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def train_and_plot_errors(self, learning_rate: float, epochs: int):
        raise NotImplementedError

    def _load_dataset(
        self, name: Literal["MNIST"], minibatch_size: int
    ) -> tuple[DataLoader, DataLoader]:
        if name == "MNIST":
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))]
            )

            train_dataset = datasets.MNIST(
                root="data", download=True, train=True, transform=transform
            )

            train_dataloader = DataLoader(
                train_dataset, batch_size=minibatch_size, shuffle=True
            )
            test_dataloader = DataLoader(
                datasets.MNIST(
                    root="data", download=True, train=False, transform=transform
                ),
                batch_size=minibatch_size,
                shuffle=True,
            )
            return train_dataloader, test_dataloader
        else:
            raise ValueError("Unsupported dataset")

    def _plot_errors(self):
        plt.plot(self.train_errors, label=f"{self.__class__} Train Error")
        plt.plot(self.test_errors, label=f"{self.__class__} Test Error")
        plt.xlabel("Epochs")
        plt.ylabel("Error Rate (%)")
        plt.legend()

    def _run_supervised_epoch(
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
            data, target = data.to(device), target.to(device)
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

    def _train_supervised(
        self,
        learning_rate: float,
        epochs: int,
    ):
        print("Starting Supervised Learning Phase")
        self.optimizer = Adam(self.parameters(), lr=learning_rate)
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            train_error = self._run_supervised_epoch(training=True)
            self.train_errors.append(train_error)
            test_error = self._run_supervised_epoch(training=False)
            self.test_errors.append(test_error)
        print("Supervised Learning Phase Complete")


# Define a traditional single-layer neural network
class BPClassifier(Classifier):
    def __init__(
        self, dataset_name: Literal["MNIST"], minibatch_size: int, hidden_size: int
    ):
        super().__init__(
            dataset_name=dataset_name,
            minibatch_size=minibatch_size,
            hidden_size=hidden_size,
        )

        # Define the neural network architecture
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

    def train_and_plot_errors(self, learning_rate: float, epochs: int):
        self._train_supervised(learning_rate=learning_rate, epochs=epochs)
        self._plot_errors()



if __name__ == "__main__":
    model = BPClassifier("MNIST", minibatch_size=64, hidden_size=2000).to(device)
    model.train_and_plot_errors(learning_rate=0.001, epochs=100)

