import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from elastic_resnet.utils import progress_bar

# based on https://github.com/kuangliu/pytorch-cifar


class Trainer:
    def __init__(
        self,
        device: torch.device,
        net: torch.nn.Module,
        lr: float = 0.1,
        max_iterations: int = 200,
        checkpoint_dir: Path = Path("./checkpoint"),
        batch_size: int = 128,
        num_workers: int = 2,
    ):
        self.net = net
        self.device = device
        self.lr = lr
        self.max_iterations = max_iterations
        self.checkpoint_dir = checkpoint_dir
        self.batch_size = batch_size

        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        trainset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform_train
        )
        self.trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

        testset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform_test
        )
        self.testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        self.reset()

    def reset(self):
        self.start_epoch = 0
        self.best_acc = 0
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.net.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4
        )
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     self.optimizer, T_max=self.max_iterations
        # )

    def resume_checkpoint(self):
        # Load checkpoint.
        print("==> Resuming from checkpoint..")
        assert os.path.isdir(
            self.checkpoint_dir
        ), "Error: no checkpoint directory found!"
        checkpoint = torch.load(self.checkpoint_dir / "ckpt.pth")
        self.net.load_state_dict(checkpoint["net"])
        self.best_acc = checkpoint["acc"]
        self.start_epoch = checkpoint["epoch"]

    def run(self):
        self.net.to(self.device)

        for epoch in range(self.start_epoch, self.start_epoch + 200):
            self.train_epoch(epoch)
            self.test_epoch(epoch)
            # self.scheduler.step()

    def train_epoch(self, epoch):
        print("\nEpoch: %d" % epoch)
        self.net.train()
        train_loss = 0
        correct = 0
        total = 0
        total_per_epoch = len(self.trainloader) * self.batch_size

        with tqdm(
            total=total_per_epoch,
            desc=f"Epoch {epoch + 1}",
            unit="img",
        ) as pbar:
            total_seen = 0
            for (inputs, targets) in self.trainloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                criterion_loss = self.criterion(outputs, targets)
                regularization_loss = self.regularization_loss()
                loss = criterion_loss + regularization_loss
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                pbar.set_postfix(
                    **{
                        "Train Loss": train_loss,
                        "Reg loss": regularization_loss.item(),
                        "Train Acc": correct / total,
                        **self.extra_training_postfix(),
                    }
                )
                pbar.update(inputs.shape[0])
                total_seen += inputs.shape[0]
                self.post_training_loop(epoch * total_per_epoch + total_seen)

    def extra_training_postfix(self):
        # overwrite me in subclasses
        return {}

    def regularization_loss(self):
        # overwrite me in subclasses
        return torch.tensor(0.0, device=self.device)

    def post_training_loop(self, total_seen: int):
        # overwrite me in subclasses
        pass

    def test_epoch(self, epoch):
        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            total_per_epoch = len(self.testloader) * self.batch_size
            with tqdm(
                total=total_per_epoch,
                desc=f"Epoch {epoch + 1}",
                unit="img",
            ) as pbar:
                for (inputs, targets) in self.testloader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.net(inputs)
                    loss = self.criterion(outputs, targets)

                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

                    pbar.set_postfix(
                        **{"Test Loss": test_loss, "Test Acc": correct / total}
                    )
                    pbar.update(inputs.shape[0])

        # Save checkpoint.
        acc = 100.0 * correct / total
        if acc > self.best_acc:
            print("Saving..")
            state = {
                "net": self.net.state_dict(),
                "acc": acc,
                "epoch": epoch,
            }
            if not os.path.isdir(self.checkpoint_dir):
                os.mkdir(self.checkpoint_dir)
            torch.save(state, self.checkpoint_dir / "ckpt.pth")
            self.best_acc = acc


class ElasticTrainer(Trainer):
    def __init__(
        self,
        device: torch.device,
        net: torch.nn.Module,
        lr: float = 0.1,
        max_iterations: int = 200,
        checkpoint_dir: Path = Path("./checkpoint"),
        batch_size: int = 128,
        num_workers: int = 2,
        channel_penalty: float = 0.001,
        weight_penalty: float = 0.001,
        resize_net_freq: int = 1000,
    ):
        super().__init__(
            device,
            net,
            lr,
            max_iterations,
            checkpoint_dir,
            batch_size,
            num_workers,
        )

        self.channel_penalty = channel_penalty
        self.weight_penalty = weight_penalty
        self.resize_net_freq = resize_net_freq
        self.last_expansion = 0

    def regularization_loss(self):
        hidden_channels_penalty = (
            self.net.get_hidden_channels_penalty() * self.channel_penalty
        )
        conv_net_weight_penalty = (
            self.net.get_conv_weight_penalty() * self.weight_penalty
        )
        return hidden_channels_penalty + conv_net_weight_penalty

    def post_training_loop(self, total_seen: int):
        if total_seen - self.last_expansion > self.resize_net_freq:
            self.last_expansion = total_seen
            resize_occcurred = self.net.resize()
            if resize_occcurred:
                # need to re-init the optimizer since the parameters are different
                self.optimizer = optim.SGD(
                    self.net.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4
                )

    def extra_training_postfix(self):
        return {"channels": self.net.get_hidden_channels_penalty().item()}
