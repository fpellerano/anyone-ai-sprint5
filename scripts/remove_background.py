"""
This script will be used to remove noisy background from cars images to
improve the quality of our data and get a better model.
The main idea is to use a vehicle detector to extract the car
from the picture, getting rid of all the background, which may cause
confusion to our CNN model.
We must create a new folder to store this new dataset, following exactly the
same directory structure with its subfolders but with new images.
"""
import argparse
import os
import cv2
from utils.utils import walkdir
from utils.detection import get_vehicle_coordinates
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Train your model.")
    parser.add_argument(
        "data_folder",
        type=str,
        help=(
            "Full path to the directory having all the cars images. Already "
            "splitted in train/test sets. E.g. "
            "`/home/app/src/data/car_ims_v1/`."
            "data/car_ims_v1"
        ),
    )
    parser.add_argument(
        "output_data_folder",
        type=str,
        help=(
            "Full path to the directory in which we will store the resulting "
            "cropped pictures. E.g. `/home/app/src/data/car_ims_v2/`."
            "data/car_ims_v2"
        ),
    )

    args = parser.parse_args()

    return args


def main(data_folder, output_data_folder):
    """
    Parameters
    ----------
    data_folder : str
        Full path to train/test images folder.

    output_data_folder : str
        Full path to the directory in which we will store the resulting
        cropped images.
    """
    if not os.path.exists(output_data_folder):
        os.makedirs(output_data_folder)
    train_path = os.path.join(output_data_folder, "train")
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    test_path = os.path.join(output_data_folder, "test")
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    # For this function, you must:
    #   1. Iterate over each image in `data_folder`, you can
    #      use Python `os.walk()` or `utils.waldir()``
                    
    #   2. Load the image

    #   3. Run the detector and get the vehicle coordinates, use
    #      utils.detection.get_vehicle_coordinates() for this task

    #   4. Extract the car from the image and store it in
    #      `output_data_folder` with the same image name. You may also need
    #      to create additional subfolders following the original
    #      `data_folder` structure.

    for dirpath, filename in tqdm(walkdir(data_folder)):
        files_to_process = os.path.join(dirpath, filename)
        im = cv2.imread(files_to_process)
        coord = get_vehicle_coordinates(im)
        im_cropped = im[coord[1] : coord[3], coord[0] : coord[2]]

        label = os.path.basename(os.path.normpath(dirpath))
        print(f"\nlabel: {label}")
        dataset_type = os.path.dirname(dirpath)
        data_type2 = os.path.dirname(dataset_type)
        dataset_type = os.path.basename(os.path.normpath(dataset_type))
        print(f"dataset type: {dataset_type}")

        if dataset_type == "train":
            label_train_path = os.path.join(train_path, label)
            if not os.path.exists(label_train_path):
                os.makedirs(label_train_path)
            image_cropped_path = os.path.join(label_train_path, filename)
            cv2.imwrite(image_cropped_path, im_cropped)
        elif dataset_type == "test":
            label_test_path = os.path.join(test_path, label)
            if not os.path.exists(label_test_path):
                os.makedirs(label_test_path)
            image_cropped_path = os.path.join(label_test_path, filename)
            cv2.imwrite(image_cropped_path, im_cropped)
        else:
            print("Dataset type not found")

if __name__ == "__main__":
    args = parse_args()
    main(args.data_folder, args.output_data_folder)
