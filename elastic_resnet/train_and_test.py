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
    parser.add_argument("--cp", default=0.001, type=float, help="channel penalty")
    parser.add_argument("--wp", default=0.001, type=float, help="weight penalty")
    parser.add_argument(
        "--resize_freq",
        default=1000,
        type=int,
        help="resize network every X training examples",
    )

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
            channel_penalty=args.cp,
            weight_penalty=args.wp,
            resize_net_freq=args.resize_freq,
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
