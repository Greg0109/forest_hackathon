import os
import cv2
import pandas as pd
import numpy as np
import click
import json
import collections

from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import Model
from keras.models import load_model


class NpEncoder(json.JSONEncoder):
    """_summary_

    Parameters
    ----------
    json : _type_
        _description_
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


@click.command()
@click.option('--model_path', default='model.h5')
@click.option('--predictions_path', default='predictions.json')
def main(model_path, predictions_path):
    df_test = pd.read_csv('test.csv')

    model = load_model(model_path)
    obtain_submission_file(model, df_test, predictions_path)


def load_test_dataset(df: pd.DataFrame) -> np.array:
    """
    Load test dataset from test dataframe (csv) example_path column.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    np.array
    """
    X_test = []
    for _, row in tqdm(df.iterrows()):
        image = cv2.imread(row.example_path)
        X_test.append(image)
    return np.array(X_test)


def obtain_submission_file(
        model: Model,
        df_test: pd.DataFrame,
        predictions_path: str):
    """
    Obtains data for submission

    Parameters
    ----------
    model : Model
    df_test : pd.DataFrame
    """
    images, image_names = [], []
    for index, row in df_test.iterrows():
        image = cv2.imread(row.example_path)
        image_names.append(
            os.path.splitext(
                os.path.basename(
                    row.example_path))[0])
        images.append(image)
    images = np.asarray(images)

    predictions = model.predict(images)
    predictions = np.argmax(predictions, axis=1)
    index = range(len(predictions))
    preds2json(index, predictions, predictions_path)


def preds2json(names: np.array, predictions: np.array, outfile: str):
    """
    Generate json for submission

    Parameters
    ----------
    names : np.array
    predictions : np.array
    outfile : str
    """
    content = dict(zip(names, predictions))
    output_json = {
        'target': content
    }
    with open('predictions.json', 'w') as outfile:
        json.dump(output_json, outfile, cls=NpEncoder)


if __name__ == '__main__':
    main()
