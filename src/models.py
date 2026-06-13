import tensorflow as tf

from src.config import IMAGE_SIZE


@tf.keras.utils.register_keras_serializable(package="MedicalImageClassification")
def resnet50_preprocess(image: tf.Tensor) -> tf.Tensor:
    return tf.keras.applications.resnet50.preprocess_input(image * 255.0)


@tf.keras.utils.register_keras_serializable(package="MedicalImageClassification")
def efficientnetb0_preprocess(image: tf.Tensor) -> tf.Tensor:
    return tf.keras.applications.efficientnet.preprocess_input(image * 255.0)


def build_custom_cnn() -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="input_image")

    x = tf.keras.layers.Conv2D(32, 3, padding="same", use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(128, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="pneumonia_probability")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="custom_cnn_baseline")


def build_resnet50_transfer() -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="input_image")
    x = tf.keras.layers.Lambda(resnet50_preprocess, name="resnet50_preprocess_input")(inputs)
    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMAGE_SIZE, 3),
        pooling=None,
        name="resnet50",
    )
    base_model.trainable = False

    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.Dense(256, activation="relu", name="classification_dense")(x)
    x = tf.keras.layers.Dropout(0.4, name="classification_dropout")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="pneumonia_probability")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="resnet50_transfer")
    return model


def set_resnet50_fine_tuning(model: tf.keras.Model, train_from_layer_name: str = "conv5_block1_out") -> None:
    base_model = model.get_layer("resnet50")
    base_model.trainable = True

    trainable = False
    for layer in base_model.layers:
        if layer.name == train_from_layer_name:
            trainable = True
        layer.trainable = trainable

    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False


def build_efficientnetb0_transfer() -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="input_image")
    x = tf.keras.layers.Lambda(efficientnetb0_preprocess, name="efficientnetb0_preprocess_input")(inputs)
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMAGE_SIZE, 3),
        pooling=None,
        name="efficientnetb0",
    )
    base_model.trainable = False

    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.Dense(256, activation="relu", name="classification_dense")(x)
    x = tf.keras.layers.Dropout(0.4, name="classification_dropout")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="pneumonia_probability")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="efficientnetb0_transfer")


def set_efficientnetb0_fine_tuning(model: tf.keras.Model, train_from_layer_name: str = "block6a_expand_conv") -> None:
    base_model = model.get_layer("efficientnetb0")
    base_model.trainable = True

    trainable = False
    for layer in base_model.layers:
        if layer.name == train_from_layer_name:
            trainable = True
        layer.trainable = trainable

    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
