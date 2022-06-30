from argparse import ArgumentParser

from monai.networks.nets import UNet
from pytorch_lightning import Trainer

from datamodules.single_step_datamodule import SingleStepDataModule
from segmentor import Segmentor


def main(params):
    base_model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=14,
        channels=(8, 16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2, 2),
    )

    checkpoint_path = (
        "checkpoints/unet-l6-s8-256-newloss-dataaug-epoch=51-val/loss=0.49.ckpt"
    )

    print("Using checkpoint:", checkpoint_path)

    model = Segmentor.load_from_checkpoint(
        checkpoint_path, model=base_model, sw_batch_size=16, sw_overlap=0.25
    )

    dm = SingleStepDataModule(
        num_labels_with_bg=14,
        supervised_dir="/mnt/HDD2/flare2022/datasets/FLARE2022/Training/FLARE22_LabeledCase50",
        val_ratio=0.2,
        predict_dir=params.predict_dir,
        output_dir=params.output_dir,
        roi_size=(256, 256, 64),
        max_workers=4,
        batch_size=8,
    )

    trainer = Trainer(logger=False, accelerator="gpu", gpus=[params.gpu], max_epochs=-1)

    trainer.validate(model, datamodule=dm)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--predict_dir", default="inputs", type=str)
    parser.add_argument("--output_dir", default="outputs", type=str)
    parser.add_argument("--gpu", default=1, type=int)
    args = parser.parse_args()

    main(args)
