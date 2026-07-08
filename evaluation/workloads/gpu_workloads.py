"""
GPU-intensive workloads for power profiling.

This module contains various GPU-bound tasks that can be used to test
the power agents's ability to measure GPU power consumption.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger


# Define a simple model with one linear layer
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        # should be around 8000MB in size
        self.linear = nn.Linear(1024 * 1024 * 100, 1)

    def forward(self, x):
        return self.linear(x)


def train_stresstest(
    num_epochs: int,
    num_batches: int,
    batch_size: int,
    device: torch.device,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
):
    for epoch in range(num_epochs):
        for batch in range(num_batches):
            # Generate random input and target data on the same device as the model
            inputs = torch.randn(batch_size, 1024 * 1024 * 100, device=device)
            targets = torch.randn(batch_size, 1, device=device)

            optimizer.zero_grad()  # Reset gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, targets)  # Compute loss

            loss.backward()  # Backward pass
            optimizer.step()  # Update parameters

            if batch % 10 == 0:
                logger.info(f"Epoch {epoch + 1}, Batch {batch}/{num_batches}, Loss: {loss.item()}")


if __name__ == "__main__":
    # Training parameters
    num_epochs = 10
    num_batches = 500
    batch_size = 8

    # Device configuration: Use GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"GPU-Test: Using device: {device}")
    model = SimpleModel().to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Start training
    logger.info("Starting GPU stress test...")
    train_stresstest(
        num_epochs=num_epochs,
        num_batches=num_batches,
        batch_size=batch_size,
        device=device,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
    )
