import os
from datetime import datetime
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
                [
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x.view(-1)),
                ]
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

    def _plot_errors(self, classifier_name: str):
        plt.plot(self.train_errors, label=f"{classifier_name} Train Error")
        plt.plot(self.test_errors, label=f"{classifier_name} Test Error")
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

    def _save(self, classifier_name: str):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join("data", "weights", classifier_name.lower(), timestamp)
        os.makedirs(save_dir, exist_ok=True)
        torch.save(
            self.state_dict(), os.path.join(save_dir, f"{classifier_name.lower()}.pth")
        )


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
        self._save("BP")


# Define the BioClassifier with Slow Implementation
class BioClassifier(Classifier):
    def __init__(
        self,
        dataset_name: Literal["MNIST"],
        minibatch_size: int,
        hidden_size: int,
        slow: bool,
        p: int,
        delta: float,
        R: float,
        h_star: float,
        tau_L: float,
        w_inh: float,
        k: Optional[int] = None,
    ):
        """Initialize the BioClassifier model.

        Parameters
        ----------
        dataset_name : Literal["MNIST"]
            Name of the dataset to use.
        minibatch_size : int
            Size of minibatches.
            NOTE: For the slow exact implementation, this parameter is ignored and updates are processed 1 at a time.
        hidden_size : int
            Size of the hidden layer
        slow : bool
            Whether to use the slow exact implementation or the fast approximate implementation.
        p : int
            Dimension of the Lebesgue norm
        delta : float
            Anti-Hebbian learning parameter
        R : float
            Radius of the sphere for normalized weights
        h_star : float
            Threshold activity for Hebbian learning
        tau_L : float
            Time constant for the lateral inhibition dynamics
        w_inh : float
            Strength of global lateral inhibition
        k : Optional[int]
            Number of top active neurons to consider for the fast implementation.
        """
        super().__init__(
            dataset_name=dataset_name,
            minibatch_size=minibatch_size,
            hidden_size=hidden_size,
        )

        # Store hyperparameters
        self.slow = slow
        self.k = k
        self.p = p
        self.delta = delta
        self.R = R
        self.h_star = h_star
        self.w_inh = w_inh
        self.tau_L = tau_L

        # Initialize W with random values from a normal distribution
        self.unsupervised_weights = torch.randn(self.hidden_size, self.input_size)

        # Initialize top supervised layer (hidden_size x output_size)
        self.supervised_weights = nn.Linear(self.hidden_size, self.output_size)

        # Define loss and optimizer for supervised phase
        self.optimizer = Adam(self.supervised_weights.parameters(), lr=0.001)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.unsupervised_weights @ x
        x = self.relu(x)
        x = self.supervised_weights(x)
        return x

    def train_and_plot_errors(
        self,
        unsupervised_lr: float,
        unsupervised_epochs: int,
        supervised_lr: float,
        supervised_epochs: int,
    ):
        """
        Train the supervised top layer and plot error rates.
        """
        self._train_unsupervised(
            learning_rate=unsupervised_lr, epochs=unsupervised_epochs
        )
        self._train_supervised(learning_rate=supervised_lr, epochs=supervised_epochs)
        if self.slow:
            self._save("bio_slow")
            self._plot_errors("bio_slow")
        else:
            self._save("bio_fast")
            self._plot_errors("bio_fast")

    def _train_unsupervised(self, learning_rate: float, epochs: int):
        """
        Perform the unsupervised learning phase.
        """
        print("Starting Unsupervised Learning Phase")
        learning_rate_update = learning_rate / epochs
        for epoch in range(epochs):
            print(f"Unsupervised Epoch {epoch+1}/{epochs}")
            for i, (data, _) in enumerate(self.train_data_loader):
                input = data.squeeze(0)

                # Solve lateral inhibition dynamics to get steady state activations
                steady_state_h = self._steady_state_activations(input)

                print(f"Activations: {steady_state_h}")

                # Update W using plasticity rule
                weight_update = self._plasticity_rule(input, steady_state_h)

                # Normalize the weight update (referenced in original implementation)
                weight_update = weight_update / weight_update.max()

                self.unsupervised_weights += learning_rate * weight_update

                print(f"Updated Weights: {self.unsupervised_weights}")

                print(f"Data {i+1}/{len(self.train_data_loader)} processed.")

            print("Epoch completed.\n")

            # Update the unsupervised learning rate
            learning_rate -= learning_rate_update

        print("Unsupervised Learning Phase Complete")

    def _steady_state_activations(self, input: torch.Tensor) -> torch.Tensor:
        """
        Compute the steady state activations h for a given input.
        """
        input_currents = (self.unsupervised_weights @ input).numpy()

        if self.slow:

            def relu(x: np.ndarray):
                return np.maximum(x, 0)

            def lateral_inhibition_dynamics(t: int, h: np.ndarray):
                inhibition = self.w_inh * np.sum(relu(h)) - (self.w_inh * relu(h))
                dhdt = (input_currents - inhibition - h) / self.tau_L
                return dhdt

            """def steady_state_event(t: int, h: np.ndarray):
                EPSILON = 1e-1
                dhdt = lateral_inhibition_dynamics(t, h)
                return np.linalg.norm(dhdt) - EPSILON  # Stop when norm < epsilon

            steady_state_event.terminal = True
            steady_state_event.direction = -1"""

            solution = solve_ivp(
                lateral_inhibition_dynamics,
                t_span=(0, 200),
                y0=input_currents,
                # Increase tolerance for faster computation
                rtol=1e-1,
                atol=1e-3,
                # events=steady_state_event,
            )

            if solution.success:
                return torch.tensor(solution.y[:, -1])
            else:
                raise RuntimeError("Steady state not reached.")
        else:
            return input_currents

    def _plasticity_rule(self, input: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Compute the synaptic weights update W based on hidden activations h and input v.
        """
        # Compute g(h)
        if self.slow:
            g = torch.where(
                h >= self.h_star,
                torch.ones_like(h),
                torch.where(h >= 0, -torch.tensor(self.delta), torch.zeros_like(h)),
            )
        else:
            _, indices = torch.topk(input, self.k + 1)
            g = torch.zeros_like(h)
            g[indices] = -self.delta
            g[indices[0]] = 1

        # Step 1: compute |W|^(p-2)
        absW = self.unsupervised_weights.abs().pow(
            self.p - 2
        )  # shape (hidden_size, input_size)

        # Step 2: p-dot product for each hidden unit
        # For each row in unsupervised_weights, do elementwise multiply by absW row, multiply by input, sum
        p_dot = (absW * self.unsupervised_weights * input).sum(
            dim=1
        )  # shape (hidden_size,)

        # Step 3: form the bracket: R^p * (absW * input) - p_dot[:, None] * W
        bracket = (self.R**self.p) * (absW * input) - p_dot.unsqueeze(
            1
        ) * self.unsupervised_weights

        # Step 4: multiply each row by g(h) (which has shape (hidden_size,))
        return g.unsqueeze(1) * bracket


if __name__ == "__main__":
    for slow in [True, False]:
        # Train a bio-inspired classifier with slow or fast implementation

        bio_model = BioClassifier(
            "MNIST",
            minibatch_size=100,
            hidden_size=2000,
            slow=slow,
            p=3,
            delta=0.4,
            R=1,
            h_star=0.5,
            tau_L=1,
            w_inh=0.1,
            k=7,
        )
        bio_model.train_and_plot_errors(
            unsupervised_epochs=1000,
            unsupervised_lr=0.04,
            supervised_lr=0.001,
            supervised_epochs=100,
        )

    # Train a traditional neural network
    bp_model = BPClassifier("MNIST", minibatch_size=64, hidden_size=2000)
    bp_model.train_and_plot_errors(learning_rate=0.001, epochs=100)

    os.makedirs("plots", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join("plots", f"errors_{timestamp}.png")
    plt.savefig(save_path)
