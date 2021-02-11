import sys
import requests
import os
import json
import collections
import pandas as pd
import random

def generate_random_predictions_on_test_set_obj_detection(test_imgs, current_dataset_classes):
    """
    Generates a prediction dataframe for image classification based on random sampling from our available classes
    """
    
    # We just use random labels for example. Our labels have to have a bounding box, confidence and class for object detection
    # bounding boxes are defined as '<xmin>, <ymin>, <xmax>, <ymax>''
    # This would be your inferences filling this DataFrame though.
    rand_lbls = ['20, 20, 80, 80' for _ in range(len(test_imgs))]
    conf = [0.95 for _ in range(len(test_imgs))]
    classes = [random.choice(current_dataset_classes) for _ in range(len(test_imgs))]
    df = pd.DataFrame({'id': test_imgs, 'bbox': rand_lbls, 'confidence': conf, 'class': classes})
    return df

def get_test_images_and_classes(dataset_path, session_token):
    """
    Helper method to dynamically get the test labels and give us the possible classes that can be submitted
    for the current dataset
    
    Params
    ------
    
    dataset_path : Path
        The path to the `development` dataset downloads
    
    session_token : str
        Your current session token so that we can look up the current session metadata
    
    Returns
    -------
    
    Tuple[List[str], List[str]]
        The list of test image ids needed to submit a prediction and the list of class names that you can predict against
    """
    # Then we can just reference our current metadata to get our dataset name and use that in the path
    headers = {'user_secret': secret, 'session_token': session_token}
    r = requests.get(f"{url}/session_status", headers=headers)
    current_dataset = r.json()['Session_Status']['current_dataset']
    current_dataset_name = current_dataset['name']
    current_dataset_classes = current_dataset['classes']

    test_imgs_dir = dataset_path.joinpath(f"{current_dataset_name}/{current_dataset_name}_{data_type}/test")
    test_imgs = [f.name for f in test_imgs_dir.iterdir() if f.is_file()]
    return test_imgs, current_dataset_classes