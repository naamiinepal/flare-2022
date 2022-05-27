from argparse import ArgumentParser

from monai.networks.nets import UNet
from pytorch_lightning import Trainer

from datamodule import DataModule
from segmentor import Segmentor


def main(params):
    base_model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=5,
        channels=(4, 8, 16, 32, 64, 128),
        strides=(2, 2, 2, 2, 2),
    )

    checkpoint_path = "playground/checkpoints/unet-l6-s4-r0-epoch=41-val_loss=0.29.ckpt"

    model = Segmentor.load_from_checkpoint(
        checkpoint_path, model=base_model, sw_batch_size=16, sw_overlap=0.25
    )

    dm = DataModule(
        predict_dir=params.predict_dir,
        output_dir=params.output_dir,
        roi_size=(128, 128, 32),
        max_workers=4,
    )

    trainer = Trainer(logger=False, accelerator="gpu", gpus=[params.gpu], max_epochs=-1)

    trainer.predict(model, datamodule=dm)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--predict_dir", default="inputs", type=str)
    parser.add_argument("--output_dir", default="outputs", type=str)
    parser.add_argument("--gpu", default=1, type=int)
    args = parser.parse_args()

    main(args)
