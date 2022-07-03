from utils.data_aug import create_data_aug_layer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers

# es un tipado de py, si la variable weights no recibe argumento, el valor por defecto será una variable tipo stirng y será "imagenet"
# igual en imput_shape, normalemente es 224x224x3 (3 son los canales RGB) = 150528 serían los pesos totales por imagen pero 
# si el desplazamiento es espacial usa mucho menos pesos, para ello se usan filtros y los mapas de características.
# Los filtros: son las neuronas 
# features map: es el resultado del filtro

# Capas convolucionales
# Capas de agrupación -> Max Pooling > reduce las dimensiones
# capas fullyconnected, se puede tener dropout
# fc de activación softmax para problemas de clasificación multiclass, si solo fueran dos claess puede ser la fc sigmoidal
 
def create_model(
    weights: str = "imagenet",
    input_shape: tuple = (224, 224, 3),
    dropout_rate: float = 0.0,
    data_aug_layer: dict = None,
    classes: int = None,
    reg_L2: float = 0.001,
):
    """
    Creates and loads the Resnet50 model we will use for our experiments.
    Depending on the `weights` parameter, this function will return one of
    two possible keras models:
        1. weights='imagenet': Returns a model ready for performing finetuning
                               on your custom dataset using imagenet weights
                               as starting point.
        2. weights!='imagenet': Then `weights` must be a valid path to a
                                pre-trained model on our custom dataset.
                                This function will return a model that can
                                be used to get predictions on our custom task.

    See an extensive tutorial about finetuning with Keras here:
    https://www.tensorflow.org/tutorials/images/transfer_learning.

    Parameters
    ----------
    weights : str
        One of None (random initialization),
        'imagenet' (pre-training on ImageNet), or the path to the
        weights file to be loaded.

    input_shape	: tuple
        Model input image shape as (height, width, channels).
        Only needed when weights='imagenet'. Otherwise, the trained model
        already has the input shape defined and we shouldn't change it.
        Input image size cannot be no smaller than 32. E.g. (224, 224, 3)
        would be one valid value.

    dropout_rate : float
        Value used for Dropout layer to randomly set input units
        to 0 with a frequency of `dropout_rate` at each step during training
        time, which helps prevent overfitting.
        Only needed when weights='imagenet'.

    data_aug_layer : dict
        Configuration from experiment YAML file used to setup the data
        augmentation process during finetuning.
        Only needed when weights='imagenet'.

    classes : int
        Model output classes.
        Only needed when weights='imagenet'. Otherwise, the trained model
        already has the output classes number defined and we shouldn't change
        it.

    Returns
    -------
    model : keras.Model
        Loaded model either ready for performing finetuning or to start doing
        predictions.
    """
    # imagesnet es un dataset de 1 millon de imágenes y 1000 categorías, un modelo preentrenado, es usado para entrenar otros modelos,
    # tomar un modelo, para clasificar 200 tipos de autos, lo que funciona bien es tomar imagenet como modelo preentrenado
    # y fine tunniarlo, hacer un transfer learning, reentrenarlo en nuestros datos particualares entonces en el primer if
    # le digo a keras que me traiga los pesos de iamgenet
    # en keras las capas de dataugmentations las capas no están preprocesadas, solo te pide que sea rgb en el rango de 0 a 255
    # en train la función de preprocessing image_dataset_from_directory hace ese formateo

    # Create the model to be used for finetuning here!
    if weights == "imagenet": # es un caso de fine tunning, acá pregunto si los pesos son los preentrenados en imagenet
        # Define the Input layer
        # Assign it to `input` variable
        # Use keras.layers.Input(), following this requirements:
        #   1. layer dtype must be tensorflow.float32

        input = tf.keras.layers.Input(shape=input_shape,name='Input', dtype=tf.float32)

        # Create the data augmentation layers here and add to the model next
        # to the input layer
        # If no data augmentation was used, skip this

        if data_aug_layer is not None:
            data_augmentation = create_data_aug_layer(data_aug_layer)
            x = data_augmentation(input)
        else:
            x = input

        # Add a layer for preprocessing the input images values
        # E.g. change pixels interval from [0, 255] to [0, 1]
        # Resnet50 already has a preprocessing function you must use here
        # See keras.applications.resnet50.preprocess_input()
        x = keras.applications.resnet50.preprocess_input(x)

        # Create the corresponding core model using
        # keras.applications.ResNet50()
        # The model created here must follow this requirements:
        #   1. Use imagenet weights
            # weights="imagenet" que el modelo tome los pesos preentregados de imagenet
        #   2. Drop top layer (imagenet classification layer)
            # include_top=False que elimine la capa 1 del modelo de imagenet
        #   3. Use Global average pooling as model output
            # pooling='avg' que tome el AVG pooling para bajar la resolución

        base_model = keras.applications.ResNet50(   weights="imagenet", 
                                                    include_top=False, 
                                                    pooling='avg',
        )
        
        # Then, freeze the base model.
        base_model.trainable = True

        # Create the model
        x = base_model(x)
        
        # Add a single dropout layer for regularization, use
        # keras.layers.Dropout()
        # dropout es util para capas fullconected 
        # en las capas convolucionales es mas comun usar  l1l2 o bachnone (tiene variables internas que tienen promedio y desviacón estandar de los pesos que entran y salen por las capas y sirve para normalizar los pesos)
        x = keras.layers.Dropout(dropout_rate)(x)

        # Add the classification layer here, use keras.layers.Dense() and
        # `classes` parameter
        # Assign it to `outputs` variable
        # a la salida siempre siempre hay que poner la fc de act softmax para que el modelo aprenda
        #3
        outputs = keras.layers.Dense(classes, kernel_regularizer=regularizers.L2(reg_L2), activation='softmax')(x)

        # Now you have all the layers in place, create a new model
        # Use keras.Model()
        # Assign it to `model` variable

        model = keras.Model(inputs=input,outputs=outputs)
    else:
        # For this particular case we want to load our already defined and
        # finetuned model, see how to do this using keras
        # Assign it to `model` variable
        model = keras.models.load_model(weights)

        # model = Model() #no es fine tunning sinó por pesos
        # model.load_weights(weights) # hay que guardar en disco, los pesos y el modelo entrenado, archivo del tipo h5
    return model
