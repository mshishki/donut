from pathlib import Path
import json
from random import choices
from shutil import copyfile

""" Prepare SROIE2019 dataset (https://rrc.cvc.uab.es/?ch=13) for usage in Donut """
DSET_DIR = Path(__file__).parents[1] / 'dataset'
dset_orig = DSET_DIR / 'SROIE2019' / '0325updated.task2train(626p)'

# Google Drive: identically named files lead to duplicates with "(1)", "(2)", etc. appended -- remove these
# 735 images and 876 text files, must be: 626
[item.unlink() for item in dset_orig.iterdir() if item.is_file() and item.stem.endswith(")")]

images = list(dset_orig.rglob("*.jpg"))
num_images = len(images)

# Create directory to host Donut-compliant dataset
dset_donut = DSET_DIR / "sroie"
dset_donut.mkdir(parents=True, exist_ok=True)

# SROIE test folder only has labels for text localization task -- dismiss and split train dataset instead
partitions = [("validation", 0.1), ("test", 0.1), ("train", 1)]
for p, frac in partitions:
    num_images_in_subset = int(num_images*frac)
    if frac < 1 or p != "train":
        idx = sorted(choices(range(len(images)), k=num_images_in_subset), reverse=True)
        images_split = [images.pop(i) for i in idx]
    else:
        images_split = images

    print(len(images_split))

    # Create directory for each subset
    new_path = dset_donut / p
    new_path.mkdir(parents=True, exist_ok=True)

    # Copy images to the new path and combine ground truth text files in a single JSONL file
    with open(new_path / 'metadata.jsonl', 'w') as f:
        for i, sample in enumerate(images_split):

            with open(sample.with_suffix(".txt"), "r") as ground_truth:
                gt = {"gt_parse": json.load(ground_truth)}

            # Dump sample information to the file
            # {"file_name": {impath0}, "ground_truth": "{\"gt_parse\": {ground_truth_parse}, ... {other_metadata...}}"}
            line = {"file_name": sample.name, "ground_truth": json.dumps(gt)}
            f.write(json.dumps(line) + "\n")

            # Copy images to the new dataset directory
            copyfile(sample, new_path / sample.name)
