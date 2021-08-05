"""Train CIFAR10 with PyTorch."""
from pathlib import Path
from .Trainer import Trainer, ElasticTrainer
import torch
import argparse

from .models.resnet import ResNet18
from .models.elastic_resnet import ElasticResNet18


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
    parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
    parser.add_argument(
        "--resume", "-r", action="store_true", help="resume from checkpoint"
    )
    parser.add_argument("--elastic", action="store_true", help="use the elastic model")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Model
    print("==> Building model..")
    if args.elastic:
        net = ElasticResNet18()
        trainer = ElasticTrainer(
            device,
            net,
            args.lr,
            checkpoint_dir=Path("./checkpoint"),
            weight_penalty=0.001,
            channel_penalty=0.001,
            expand_net_freq=1000,
        )
    else:
        net = ResNet18()
        trainer = Trainer(
            device,
            net,
            args.lr,
            checkpoint_dir=Path("./checkpoint"),
        )
    if args.resume:
        trainer.resume_checkpoint()
    trainer.run()
