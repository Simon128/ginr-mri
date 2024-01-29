from argparse import ArgumentParser

from .trainer import Trainer

def parser():
    parser = ArgumentParser(
        prog="ginr-mri",
        description="Train and run generalizable INRs on MRI scans"
    )
    parser.add_argument("mode", choices=["train", "test"])
    parser.add_argument("config", type=str)
    parser.add_argument("--ranks", nargs="+", default=[0])
    return parser

if __name__ == "__main__":
    _parser = parser()
    args = _parser.parse_args()
    if args.mode == "train":
        trainer = Trainer(args.config)
        trainer.fit()
    else:
        # run test
        pass
