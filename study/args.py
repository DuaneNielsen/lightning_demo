import pytorch_lightning as pl
import pytorch_lightning.utilities.argparse as pl_argparse
from argparse import ArgumentParser


class ArgumentModule(pl.LightningModule):
    def __init__(self, model_name: str, lr: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()

    @classmethod
    def from_argparse_args(cls, args):
        kwargs = {}
        for name, tipe, default in pl_argparse.get_init_arguments_and_types(cls):
            kwargs[name] = vars(args)[name]
        return cls(**kwargs)

    @classmethod
    def add_argparse_args(cls, parser):
        for name, tipe, default in pl_argparse.get_init_arguments_and_types(cls):
            parser.add_argument(f'--{name}', type=tipe[0], default=default)


if __name__ == '__main__':

    model = ArgumentModule('hello_world', lr=1e-4)
    print(model.hparams)

    parser = ArgumentParser()
    ArgumentModule.add_argparse_args(parser)
    args = parser.parse_args(['--model_name', 'tarantula'])
    model = ArgumentModule.from_argparse_args(args)
    print(model.hparams)