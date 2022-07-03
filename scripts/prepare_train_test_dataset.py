"""
This script will be used to separate and copy images coming from
`car_ims.tgz` (extract the .tgz content first) between `train` and `test`
folders according to the column `subset` from `car_dataset_labels.csv`.
It will also create all the needed subfolders inside `train`/`test` in order
to copy each image to the folder corresponding to its class.

The resulting directory structure should look like this:
    data/
    ├── car_dataset_labels.csv
    ├── car_ims
    │   ├── 000001.jpg
    │   ├── 000002.jpg
    │   ├── ...
    ├── car_ims_v1
    │   ├── test
    │   │   ├── AM General Hummer SUV 2000
    │   │   │   ├── 000046.jpg
    │   │   │   ├── 000047.jpg
    │   │   │   ├── ...
    │   │   ├── Acura Integra Type R 2001
    │   │   │   ├── 000450.jpg
    │   │   │   ├── 000451.jpg
    │   │   │   ├── ...
    │   ├── train
    │   │   ├── AM General Hummer SUV 2000
    │   │   │   ├── 000001.jpg
    │   │   │   ├── 000002.jpg
    │   │   │   ├── ...
    │   │   ├── Acura Integra Type R 2001
    │   │   │   ├── 000405.jpg
    │   │   │   ├── 000406.jpg
    │   │   │   ├── ...

final:
python scripts/prepare_train_test_dataset.py /home/app/src/data/car_ims /home/app/src/src/data /home/app/src/src/data/car_ims_v1

"""

import argparse
import os
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Train your model.")
    parser.add_argument(
        "data_folder",
        type=str,
        help=(
            "Full path to the directory having all the cars images. E.g. "
            "`/home/app/src/data/car_ims/`."
            "data/car_ims/"
        ),
    )
    parser.add_argument(
        "labels",
        type=str,
        help=(
            "Full path to the CSV file with data labels. E.g. "
            "`/home/app/src/data/car_dataset_labels.csv`."
            "data/car_dataset_labels.csv" 
        ),
    )
    parser.add_argument(
        "output_data_folder",
        type=str,
        help=(
            "Full path to the directory in which we will store the resulting "
            "train/test splits. E.g. `/home/app/src/data/car_ims_v1/`."
            "data/car_ims_v1/"
        ),
    )

    args = parser.parse_args()

    return args

def main(data_folder, labels, output_data_folder):
    """
    Parameters
    ----------
    data_folder : str
        Full path to raw images folder.

    labels : str
        Full path to CSV file with data annotations.

    output_data_folder : str
        Full path to the directory in which we will store the resulting
        train/test splits.
    """
    # For this function, you must:
    #   1. Load labels CSV file
    car_dataset_labels = pd.read_csv("data/car_dataset_labels.csv")
    #   2. Iterate over each row in the CSV, create the corresponding
    #      train/test and class folders
    c_d_l_columns_names = car_dataset_labels.columns.values
    c_d_l_subset = car_dataset_labels['subset'].unique()
    c_d_l_class = car_dataset_labels['class'].unique()

    for i in c_d_l_subset:
        # print(f"data/{i}")
        newpath = f"data/car_ims_v1/{i}" 
        if not os.path.exists(newpath):
            os.makedirs(newpath)

    #   3. Copy the image to the new folder structure. We recommend you to
    #      use `os.link()` to avoid wasting disk space with duplicated files
    # TODO

    for _, line in car_dataset_labels.iterrows():
        # if line['img_name']=='016183.jpg':
        if line['subset']=='test':
            name_class_test=line['class'].replace(' ', '_')
            newpath2 = f"data/car_ims_v1/test/{name_class_test}" 
            # print(newpath2)
            os.makedirs(newpath2, exist_ok=True)
            src_test = f'data/car_ims/{line["img_name"]}'
            dst_test = f'data/car_ims_v1/test/{name_class_test}/{line["img_name"]}'
            try:
                os.link(src_test, dst_test)
            except:
                    print(f'The image {dst_test} exist in the path.') 
        else:
            if line['subset']=='train':
                name_class_train=line['class'].replace(' ', '_')
                newpath3 = f"data/car_ims_v1/train/{name_class_train}" 
                os.makedirs(newpath3, exist_ok=True)
                src_train = f'data/car_ims/{line["img_name"]}'
                dst_train = f'data/car_ims_v1/train/{name_class_train}/{line["img_name"]}'
                try:
                    os.link(src_train, dst_train)
                except:
                    print(f'The image {dst_train} exist in the path.') 


if __name__ == "__main__":
    args = parse_args()
    main(args.data_folder, args.labels, args.output_data_folder)