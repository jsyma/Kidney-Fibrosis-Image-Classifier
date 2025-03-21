import numpy as np
import tensorflow as tf
import os
import re
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import RandomizedSearchCV, GroupKFold
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from tensorflow import keras
from keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.client import device_lib
from scipy.stats import uniform, loguniform


def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]


def extract_features(name, generator, model):
    features = []
    labels = []

    try:
        features = np.load('./model_features/train_features_' + name + '.npy')
        labels = np.load('./model_features/train_labels_' + name + '.npy')
    except OSError:
        features = model.predict(generator, verbose=1)
        labels = generator.classes

        np.save('./model_features/train_features_' + name + '.npy', features)
        np.save('./model_features/train_labels_' + name + '.npy', labels)

        print('Saved model features and labels for ' + name)
    finally:
        return features, labels


print(get_available_devices())
for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

dataset_paths = {
    'pa_750': './Data/Path_Hidden/750',  # PA 750 data path
    'pa_850': './Data/Path_Hidden/850',  # PA 850 data path
    'us_750': './Data/Path_Hidden/750',  # US 750 data path
    'us_850': './Data/Path_Hidden/850'   # US 850 data path
}
# select dataset from above
dataset_name = 'us_850'
dataset_path = dataset_paths[dataset_name]

# make sure not to use the same mouse for training and validation
class_dirs = [os.path.join(dataset_path, cls) for cls in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, cls))]

# dataset storing tuples of (filepath, class, mouse_id)
data = []

# Extract mouse ID from file names and organize data
for class_dir in class_dirs:
    class_name = os.path.basename(class_dir)
    for filename in os.listdir(class_dir):
        if filename.endswith('.jpg'):
            # filename format: PA_{mouseid}_D7_left_750_{framenum}.jpg
            mouse_id = re.search('(PA|B)_(.+)_.+_.+_(750|850)_.+\.jpg', filename).group(2)
            file_path = os.path.join(class_dir, filename)
            data.append((file_path, class_name, mouse_id))

# Create DataFrame from data
df = pd.DataFrame(data, columns=['filepath', 'class', 'mouse_id'])
mouse_ids = df['mouse_id'].values

data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1)
        ]
    )
resnet50v2 = tf.keras.applications.ResNet50V2(input_shape=(512, 512, 3),
                                              include_top=False,
                                              pooling='avg',
                                              weights='imagenet')
model = resnet50v2

param_distributions = {
    'C': loguniform(1e-3, 1e3),  # Regularization strength
    'kernel': ['sigmoid'],
    'gamma': loguniform(1e-4, 1),  # Kernel coefficient for RBF
}

scoring = {
    'accuracy': 'accuracy',
    'precision': make_scorer(precision_score, average='weighted'),
    'recall': make_scorer(recall_score, average='weighted'),
    'f1': make_scorer(f1_score, average='weighted'),
}

train_datagen = ImageDataGenerator(rescale=1.0/255)
train_generator = train_datagen.flow_from_dataframe(
    df,
    x_col='filepath',
    y_col='class',
    target_size=(512, 512),
    batch_size=32,
    class_mode='sparse',
    shuffle=False
)

# use k-fold cross validation for grouped datasets (k-1/k train, 1/k val)
n_splits = 4
gkf = GroupKFold(n_splits=n_splits)

train_features, train_labels = extract_features(dataset_name, train_generator, model)

svm = SVC(probability=True)
random_search = RandomizedSearchCV(
    estimator=svm,
    param_distributions=param_distributions,
    n_iter=30,
    scoring=scoring,  # Multi-metric scoring
    refit='f1',
    cv=gkf,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

# fit with groups
random_search.fit(train_features, train_labels, groups=mouse_ids)

cv_results = random_search.cv_results_

print("Best Hyperparameters:", random_search.best_params_)
print("Best Cross-Validation Score:", random_search.best_score_)

