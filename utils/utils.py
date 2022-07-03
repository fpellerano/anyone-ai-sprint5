import os
import yaml
import tensorflow as tf
import numpy as np
from tqdm import tqdm 
from tensorflow import keras

def validate_config(config):
    """
    Takes as input the experiment configuration as a dict and checks for
    minimum acceptance requirements.

    Parameters
    ----------
    config : dict
        Experiment settings as a Python dict.
    """
    if "seed" not in config:
        raise ValueError("Missing experiment seed")

    if "data" not in config:
        raise ValueError("Missing experiment data")

    if "directory" not in config["data"]:
        raise ValueError("Missing experiment training data")


def load_config(config_file_path):
    """
    Loads experiment settings from a YAML file into a Python dict.
    See: https://pyyaml.org/.

    Parameters
    ----------
    config_file_path : str
        Full path to experiment configuration file.
        E.g: `/home/app/src/experiments/exp_001/config.yml`

    Returns
    -------
    config : dict
        Experiment settings as a Python dict.
    """

    # Write YAML file
    # with io.open('data.yaml', 'w', encoding='utf8') as outfile:
    # yaml.dump(data, outfile, default_flow_style=False, allow_unicode=True)

    # Read YAML file
    with open(config_file_path, 'r') as stream:
        config = yaml.safe_load(stream)
    #print(config) 

    # Don't remove this as will help you doing some basic checks on config
    # content
    validate_config(config)

    return config


def get_class_names(config):
    """
    It's not always easy to track how Keras maps our dataset classes to
    the model outputs.
    Given an image, the model output will be a 1-D vector with probability
    scores for each class. The challenge is, how to map our class names to
    each score in the output vector.
    We will use this function to provide a class order to Keras and keep
    consistency between training and evaluation.

    Parameters
    ----------
    config : dict
        Experiment settings as Python dict.

    Returns
    -------
    classes : list
        List of classes as string.
        E.g. ['AM General Hummer SUV 2000', 'Buick Verano Sedan 2012',
                'FIAT 500 Abarth 2012', 'Jeep Patriot SUV 2012',
                'Acura Integra Type R 2001', ...]
    """
    return sorted(os.listdir(os.path.join(config["data"]["directory"])))


def walkdir(folder):
    """
    Walk through all the files in a directory and its subfolders.

    Parameters
    ----------
    folder : str
        Path to the folder you want to walk.

    Returns
    -------
        For each file found, yields a tuple having the path to the file
        and the file name.
    """
    for dirpath, _, files in os.walk(folder):
        for filename in files:
            yield (dirpath, filename)


def predict_from_folder(folder, model, input_size, class_names):
    """
    Walk through all the image files in a directory, loads them, applies
    the corresponding pre-processing and sends to the model to get
    predictions.

    This function will also return the true label for each image, to do so,
    the folder must be structured in a way in which images for the same
    category are grouped into a folder with the corresponding class
    name. This is the same data structure as we used for training our model.

    Parameters
    ----------
    folder : str
        Path to the folder you want to process.

    model : keras.Model
        Loaded keras model.

    input_size : tuple
        Keras model input size, we must resize the image to math these
        dimensions.

    class_names : list
        List of classes as string. It allow us to map model output IDs to the
        corresponding class name, e.g. 'Jeep Patriot SUV 2012'.

    Returns
    -------
    predictions, labels : tuple
        It will return two lists:
            - predictions: having the list of predicted labels by the model.
            - labels: is the list of the true labels, we will use them to
                      compare against model predictions.
    """
    # Use keras.utils.load_img() to correctly load the image and
    # keras.utils.img_to_array() to convert it to the format needed
    # before sending it to our model.
    # You can use os.walk() or walkdir() to iterate over the files in the
    # folder.
    # Don't forget you must not return the raw model prediction. Model
    # prediction will be a vector assigning probability scores to each
    # class. You must take the position of the element in the vector with
    # the highest probability and use that to get the corresponding class
    # name from `class_names` list.

    #folder="data/car_ims_v1/test"
    pred_list = []
    class_list = []

    for dirpath, filename in tqdm(walkdir(folder)):
        #folder="data/car_ims_v1/test"
        img_path = os.path.join(dirpath, filename)

        #input_size (224, 224)
        img = keras.utils.load_img(img_path, target_size=input_size)
        img_array = keras.preprocessing.image.img_to_array(img)
        img_batch = np.expand_dims(img_array, axis=0)
        #model = tf.keras.applications.resnet50.ResNet50()
        predictions = model.predict(img_batch)

        predicted_class = class_names[np.argmax(predictions)]
        pred_list.append(predicted_class)
        _, true_label = os.path.split(dirpath)
        class_list.append(true_label)
    predictions = pred_list
    labels = class_list

    return predictions, labels