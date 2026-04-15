import torch
import torch.nn as nn
import torch.optim as optim


class OptimizedTrainer:
    """Memory optimized trainer"""

    def __init__(self, model, optimizer, criterion, accumulation_steps=4):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.accumulation_steps = accumulation_steps
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        self.step_count = 0

    def train_step(self, inputs, targets):
        """Execute a training step (supports gradient accumulation)"""

        # Mixed precision training forward pass
        with torch.cuda.amp.autocast(enabled=self.scaler is not None):
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Gradient accumulation: divide loss by accumulation steps
            loss = loss / self.accumulation_steps

        # Backward pass
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        self.step_count += 1

        # Accumulate gradients, update every accumulation_steps steps
        if self.step_count % self.accumulation_steps == 0:
            self._update_parameters()
            return loss.item() * self.accumulation_steps  # Return original loss value

        return None  # Do not return loss during accumulation steps

    def _update_parameters(self):
        """Update model parameters"""
        if self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.optimizer.zero_grad()

    def reset_step_count(self):
        """Reset step counter"""
        self.step_count = 0


def create_memory_efficient_training_loop():
    """Create a memory efficient training loop example"""

    # Configure memory optimization
    configure_memory_optimization()

    # Set up gradient accumulation
    batch_size = 4  # Reduce batch size
    accumulation_steps = 4
    gradient_accumulation_setup(batch_size, accumulation_steps)

    print("✅ Memory efficient training loop created")


# Usage example
if __name__ == "__main__":
    # Example: how to use in actual training
    create_memory_efficient_training_loop()

    # Simulate a simple model
    model = nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 128, 3, padding=1),
        nn.ReLU(),
    )

    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    # Create optimized trainer
    trainer = OptimizedTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        accumulation_steps=4
    )

    print("Optimized trainer ready to use")