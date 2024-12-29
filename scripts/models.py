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
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)


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
                train_dataset,
                batch_size=minibatch_size,
                shuffle=True,
                generator=torch.Generator(device=device),
            )
            test_dataloader = DataLoader(
                datasets.MNIST(
                    root="data",
                    download=True,
                    train=False,
                    transform=transform,
                ),
                batch_size=minibatch_size,
                shuffle=True,
                generator=torch.Generator(device=device),
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
        for data, target in tqdm(data_loader, desc="Batch:"):
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
        classifier_name: str,
    ):
        print(f"Starting Supervised Learning Phase for {classifier_name}")
        self.optimizer = Adam(self.parameters(), lr=learning_rate)
        for epoch in tqdm(range(epochs), desc="Supervised Epoch:"):
            train_error = self._run_supervised_epoch(training=True)
            self.train_errors.append(train_error)
            test_error = self._run_supervised_epoch(training=False)
            self.test_errors.append(test_error)

            # Save the model every 50 epochs
            if (epoch + 1) % 50 == 0:
                self._save(f"{classifier_name}_supervised_epoch_{epoch+1}")

        print("Supervised Learning Phase Complete")

    def _save(self, classifier_name: str):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join("data", "weights", classifier_name.lower(), timestamp)
        os.makedirs(save_dir, exist_ok=True)
        torch.save(
            self.state_dict(), os.path.join(save_dir, f"{classifier_name.lower()}.pth")
        )
        print(f"Model saved to {save_dir}")

    def load(self, path: str):
        """Load the model weights from the specified path."""
        self.load_state_dict(torch.load(path))

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

    def train_and_plot_errors(
        self, learning_rate: float, epochs: int, classifier_name: str
    ):
        self._train_supervised(
            learning_rate=learning_rate, epochs=epochs, classifier_name=classifier_name
        )
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
        # (batch_size, input_dim) @ (hidden_size, input_dim).T -> (batch_size, hidden_size)
        x = x @ self.unsupervised_weights.T
        x = self.relu(x)
        x = self.supervised_weights(x)
        x = self.softmax(x)
        return x

    def train_and_plot_errors(
        self,
        unsupervised_lr: float,
        unsupervised_epochs: int,
        supervised_lr: float,
        supervised_epochs: int,
        classifier_name: str,
    ):
        """
        Train the supervised top layer and plot error rates.
        """
        self._train_unsupervised(
            learning_rate=unsupervised_lr,
            epochs=unsupervised_epochs,
            classifier_name=classifier_name,
        )
        self._train_supervised(
            learning_rate=supervised_lr,
            epochs=supervised_epochs,
            classifier_name=classifier_name,
        )
        if self.slow:
            self._save("bio_slow")
            self._plot_errors("bio_slow")
        else:
            self._save("bio_fast")
            self._plot_errors("bio_fast")

    def _train_unsupervised(
        self, learning_rate: float, epochs: int, classifier_name: str
    ):
        """
        Perform the unsupervised learning phase.
        """
        print(
            f"Starting Unsupervised Learning Phase for BioClassifier {"slow" if self.slow else "fast"}"
        )
        learning_rate_update = learning_rate / epochs
        for epoch in tqdm(range(epochs), desc="Unsupervised Epoch:"):
            for i, (input, _) in tqdm(enumerate(self.train_data_loader), desc="Batch:"):
                input = input.to(device)  # ensure data is on the same device
                input_currents = self._compute_input_currents(input)

                # Solve lateral inhibition dynamics to get steady state activations
                steady_state_h = self._steady_state_activations(input_currents)

                # Update W using plasticity rule
                batch_updates = self._plasticity_rule(
                    input, input_currents, steady_state_h
                )

                weight_update = batch_updates.mean(dim=0)

                # Normalize the weight update (referenced in original implementation)
                weight_update /= weight_update.max()

                self.unsupervised_weights += learning_rate * weight_update

            # Save the model every 50 epochs
            if (epoch + 1) % 50 == 0:
                self._save(f"{classifier_name}_unsupervised_epoch_{epoch+1}")

            # Update the unsupervised learning rate
            learning_rate -= learning_rate_update

        print("Unsupervised Learning Phase Complete")

    def _compute_input_currents(self, input: torch.Tensor) -> torch.Tensor:
        """
        unsupervised_weights: (hidden_size, input_dim)
        input: (batch_size, input_dim)
        returns: (hidden_size, batch_size) or (batch_size, hidden_size)
        """
        # 1) compute elementwise w_abs^(p-2)
        w_abs_pow = self.unsupervised_weights.abs().pow(self.p - 2)
        # 2) multiply by W elementwise
        effective_w = w_abs_pow * self.unsupervised_weights
        # 3) matrix multiply with input^T
        #    yields shape (hidden_size, batch_size)
        currents = effective_w @ input.T
        return currents

    def _steady_state_activations(self, input_currents: torch.Tensor) -> torch.Tensor:
        """
        Compute the steady state activations h for a given input.
        """

        if self.slow:

            def scipy_steady_state(curr_input):
                def lateral_inhibition_dynamics(t, h):
                    relu_h = np.maximum(h, 0)
                    inhibition = self.w_inh * np.sum(relu_h) - self.w_inh * relu_h
                    return (curr_input - inhibition - h) / self.tau_L

                solution = solve_ivp(
                    lateral_inhibition_dynamics,
                    t_span=(0, 200),
                    y0=curr_input,
                    method="RK23",
                    t_eval=[0, 100, 200],
                )
                return solution.y[:, -1]

            solutions = []

            for i in range(input_currents.shape[1]):
                # SciPy solution on the same column i
                solutions.append(scipy_steady_state(input_currents[:, i].numpy()))
            return torch.tensor(np.array(solutions), device=device)
        else:
            return input_currents

    def _plasticity_rule(
        self, input: torch.Tensor, input_currents: torch.Tensor, h: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the synaptic weights update W based on hidden activations h and input v.
        """
        # Compute g(h)
        if self.slow:
            g = torch.where(
                h >= self.h_star,
                torch.ones_like(h),
                torch.where(h >= 0, -torch.tensor(self.delta), torch.zeros_like(h)),
            ).T
        else:
            _, indices = torch.topk(h, self.k + 1)
            g = torch.zeros_like(h)
            g[indices] = -self.delta
            g[indices[0]] = 1

        # Step 3: form the bracket: R^p * (absW * input) - p_dot[:, None] * W
        # bracket => [H, B, D] = [2000, 100, 784]
        # input_currents.shape => [H, B] -> unsqueeze(2) => [H, B, 1]
        # unsupervised_weights.shape => [H, D] -> unsqueeze(1) => [H, 1, D]
        bracket = (self.R**self.p) * (input) - input_currents.unsqueeze(
            2
        ) * self.unsupervised_weights.unsqueeze(1)

        # g.shape => [B, H]
        # to broadcast over the last dimension D, unsqueeze dim=2 => [H, B, 1]
        weight_update = bracket * g.unsqueeze(2)

        # Average over the batch dimension
        weight_update = weight_update.mean(dim=1)
        return weight_update
