"""
This script will be used for training our CNN. The only inputaugmentation argument it
should receive is the path to our configuration file in which we define all
the experiment settings like dataset, model output folder, epochs,
learning rate, data augmentation, etc.
"""
import argparse

import tensorflow as tf
from tensorflow import keras

from models import resnet_50
from utils import utils

# Prevent tensorflow to allocate the entire GPU
# https://www.tensorflow.org/api_docs/python/tf/config/experimental/set_memory_growth
physical_devices = tf.config.list_physical_devices("GPU")
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

# Supported optimizer algorithms
OPTIMIZERS = {
    "adam": keras.optimizers.Adam,
    "sgd": keras.optimizers.SGD,
}


# Supported callbacks: es una clase que se aplica despues de cada ephoc. 
# EarlyStopping:Se usa mucho cuando la loss en validations deja de ascender detiene el entrenamiento (overfiting) 
# TensorBoard: guarda los log para depues ver los resultdos en vivo
# ModelCheckpoint: guarda el estado de los pesos, de los 50 ephocs la 12 dio los mejores resultaods, podemos retrotaer.

CALLBACKS = {
    "model_checkpoint": keras.callbacks.ModelCheckpoint,
    "tensor_board": keras.callbacks.TensorBoard,
    "early_stopping": keras.callbacks.EarlyStopping,
}


def parse_args():
    """
    Use argparse to get the input parameters for training the model.
    """
    parser = argparse.ArgumentParser(description="Train your model.")
    parser.add_argument(
        "config_file",
        type=str,
        help="Full path to experiment configuration file.",
    )

    args = parser.parse_args()
    return args


def parse_optimizer(config):
    """
    Get experiment settings for optimizer algorithm.

    Parameters
    ----------
    config : str
        Experiment settings.
    """
    opt_name, opt_params = list(config["compile"]["optimizer"].items())[0]
    optimizer = OPTIMIZERS[opt_name](**opt_params)

    del config["compile"]["optimizer"]

    return optimizer


def parse_callbacks(config):
    """
    Add Keras callbacks based on experiment settings.

    Parameters
    ----------
    config : str
        Experiment settings.
    """
    callbacks = []
    if "callbacks" in config["fit"]:
        for callbk_name, callbk_params in config["fit"]["callbacks"].items():
            callbacks.append(CALLBACKS[callbk_name](**callbk_params))
        del config["fit"]["callbacks"]

    return callbacks
            
"""
    CALLBACKS[callbk_name] = keras.callbacks.ModelCheckpoint si el callbk_name = "model_checkpoint" entonces
    CALLBACKS["model_checkpoint"] = keras.callbacks.ModelCheckpoint
    
    CALLBACKS["model_checkpoint"](**callbk_params) aca es como si estoy creando el objeto a partir de la clase, 
    instanaciandolo y con el doble asterisco le paso todos los 

        callbk_params={    
            "filepath": "/home/app/src/experiments/exp_001/model.{epoch:02d}-{val_loss:.4f}.h5"
            "save_best_only": True
            }
    Entonces, hacer: 

    CALLBACKS[callbk_name](**callbk_params)
    
    es lo smiso que hacer: 

    keras.callbacks.ModelCheckpoint(
        filepath: "/home/app/src/experiments/exp_001/model.{epoch:02d}-{val_loss:.4f}.h5"
        save_best_only: True

    )
"""

def main(config_file):
    """
    Code for the training logic.

    Parameters
    ----------
    config_file : str
        Full path to experiment configuration file.
    """
    # Load configuration file, use utils.load_config()
    config = utils.load_config(config_file)

    # Get the list of output classes
    # We will use it to control the order of the output predictions from
    # keras is consistent
    class_names = utils.get_class_names(config)

    # Check if number of classes is correct
    if len(class_names) != config["model"]["classes"]:
        raise ValueError(
            "The number classes between your dataset and your model"
            "doen't match."
        )

    # Load training dataset
    # We will split train data in train/validation while training our
    # model, keeping away from our experiments the testing dataset
    train_ds = keras.preprocessing.image_dataset_from_directory(
        subset="training",
        class_names=class_names,
        seed=config["seed"],
        **config["data"],
    )
    val_ds = keras.preprocessing.image_dataset_from_directory(
        subset="validation",
        class_names=class_names,
        seed=config["seed"],
        **config["data"],
    )

    # Creates a Resnet50 model for finetuning
    cnn_model = resnet_50.create_model(**config["model"])
    print(cnn_model.summary())

    # Compile model, prepare for training
    optimizer = parse_optimizer(config)
    cnn_model.compile(
        optimizer=optimizer,
        **config["compile"],
    )

    # Start training!
    # los dos asteriscos es que busca todos los parametros que tiene config y hace un and pathing
    callbacks = parse_callbacks(config)
    cnn_model.fit(
        train_ds, validation_data=val_ds, callbacks=callbacks, **config["fit"]
    )


if __name__ == "__main__":
    args = parse_args()
    main(args.config_file)
