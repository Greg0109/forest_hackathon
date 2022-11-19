
import click
import cv2

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50V2

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


@click.command()
@click.option('--logdir', default='./logs/')
def main(logdir):
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    df_train = pd.read_csv('train.csv')
    X_train, X_val, y_train, y_val = load_train_val_dataset(df_train)

    model = generate_model(verbose=True)

    callback_metric = tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch,
                                                        logs: print(custom_metric_callback(y_val, model.predict(X_val))))
    model.fit(X_train, y_train, epochs=15, batch_size=32,
              validation_data=(X_val, y_val),
              callbacks=[callback_metric, tensorboard_callback])

    model.save('model.h5')


def load_train_val_dataset(df: pd.DataFrame, test_size: float = 0.25, seed: int = 69420) -> tuple:
    """
    Load train and validation dataset from training dataframe (csv) example_path column.

    Parameters
    ----------
    df : pd.DataFrame
        Train dataframe
    test_size : float, optional
        Percentage of validation data to be used, by default 0.25
    seed : int, optional
        by default 69420

    Returns
    -------
    tuple
        X_train, X_test, y_train, y_test
    """
    data = []
    for _, row in df.iterrows():
        image = cv2.imread(row.example_path)
        data.append(image)
    data = np.array(data)
    labels = to_categorical(np.asarray(df.label))

    X_train, X_test, y_train, y_test = train_test_split(data, labels,
                                                        test_size=test_size,
                                                        random_state=seed)

    return X_train, X_test, y_train, y_test


def generate_model(verbose: bool = False) -> Model:
    """
    Generate pre-trained model for training.

    Parameters
    ----------
    verbose : bool, optional
        Print summary of the model, by default False

    Returns
    -------
    Model
        Return Model
    """

    base_model = ResNet50(input_shape=(332, 332, 3),
                           include_top=False,
                           weights='imagenet')
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(3, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=x)

    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

    if verbose:
        model.summary()

    return model

def custom_metric_callback(y_true: np.array, y_pred: np.array) -> float:
    """
    Calculate Callback function metrics for F1 score

    Parameters
    ----------
    y_true : np.array
        Predictions
    y_pred : np.array
        Labels
    Returns
    -------
    float
        _description_
    """
    y_true = tf.argmax(y_true, axis=1)
    y_pred = tf.argmax(y_pred, axis=1)
    return f1_score(y_true, y_pred, average='macro')

if __name__ == '__main__':
    main()
