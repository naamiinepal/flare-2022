import os
from argparse import ArgumentParser
from glob import glob

import lmdb

BASE_DIR = "/mnt/HDD2/safal/EfficientSegmentation/FlareSeg/dataset"

value_template = {
    "image_path": "",
    "mask_path": "",
    "smooth_mask_path": "",
    "coarse_image_path": "",
    "coarse_mask_path": "",
    "fine_image_path": "",
    "fine_mask_path": "",
}


def read_txt(txt_file):
    txt_lines = []
    with open(txt_file, "r") as f:
        for line in f.readlines():
            line = line.strip()
            txt_lines.append(line)
    return txt_lines


def get_file_paths(base_dir, dir_name, filetype=""):
    file_paths = glob(os.path.join(base_dir, dir_name, f"*{filetype}"))
    file_paths.sort()
    return file_paths


def create_lmdb(args):
    series_ids = read_txt(
        os.path.join(args.base_dir, "file_list", f"{args.stage}_series_uids.txt")
    )
    image_paths = get_file_paths(
        base_dir=args.base_dir, dir_name=f"{args.stage}_images", filetype=".nii.gz"
    )
    if args.stage == "train":
        mask_paths = get_file_paths(
            base_dir=args.base_dir, dir_name=f"{args.stage}_mask", filetype=".nii.gz"
        )
    else:
        mask_paths = [""] * len(image_paths)
    smooth_mask_paths = [""] * len(image_paths)
    coarse_image_paths = get_file_paths(
        base_dir=args.base_dir, dir_name="coarse_image/160_160_160"
    )
    coarse_mask_paths = get_file_paths(
        base_dir=args.base_dir, dir_name="coarse_mask/160_160_160"
    )
    fine_image_paths = get_file_paths(
        base_dir=args.base_dir, dir_name="fine_image/192_192_192"
    )
    fine_mask_paths = get_file_paths(
        base_dir=args.base_dir, dir_name="fine_mask/192_192_192"
    )
    env = lmdb.open(args.stage, map_size=int(1e9))
    txn = env.begin(write=True)
    for (
        series_id,
        img_path,
        mask_path,
        smooth_mask_path,
        coarse_img_path,
        coarse_mask_path,
        fine_img_path,
        fine_mask_path,
    ) in zip(
        series_ids,
        image_paths,
        mask_paths,
        smooth_mask_paths,
        coarse_image_paths,
        coarse_mask_paths,
        fine_image_paths,
        fine_mask_paths,
    ):
        key = f"{series_id}".encode()
        value = value_template.copy()
        value["image_path"] = img_path
        value["mask_path"] = mask_path
        value["smooth_mask_path"] = smooth_mask_path
        value["coarse_image_path"] = coarse_img_path
        value["coarse_mask_path"] = coarse_mask_path
        value["fine_image_path"] = fine_img_path
        value["fine_mask_path"] = fine_mask_path
        txn.put(key, str(value).encode())
    txn.commit()
    env.close()

    for (
        series_id,
        img_path,
        mask_path,
        smooth_mask_path,
        coarse_img_path,
        coarse_mask_path,
        fine_img_path,
        fine_mask_path,
    ) in zip(
        series_ids,
        image_paths,
        mask_paths,
        smooth_mask_paths,
        coarse_image_paths,
        coarse_mask_paths,
        fine_image_paths,
        fine_mask_paths,
    ):
        print(
            series_id,
            img_path,
            mask_path,
            smooth_mask_path,
            coarse_img_path,
            coarse_mask_path,
            fine_img_path,
            fine_mask_path,
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--base_dir", type=str, default=BASE_DIR)
    parser.add_argument("--stage", type=str, default="train")

    args = parser.parse_args()
    create_lmdb(args)
