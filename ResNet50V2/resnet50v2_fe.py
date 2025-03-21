import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import re
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.client import device_lib


def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]


def extract_features(generator, model):
    features = model.predict(generator, verbose=1)
    labels = generator.classes  # Corresponding labels
    return features, labels


def plot_multiple_confusion_matrices(conf_matrices, class_names, titles, figsize=(12, 8)):
    n = len(conf_matrices)
    rows = int(np.ceil(n / 2))
    cols = 2 if n > 1 else 1

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if n > 1 else [axes]

    for i, cm in enumerate(conf_matrices):
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(ax=axes[i], cmap=plt.cm.Blues, colorbar=False)
        axes[i].set_title(titles[i])

    # Hide any unused subplots
    for j in range(len(conf_matrices), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


print(get_available_devices())
for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)


dataset_paths = {
    'pa_750': './Data/Path_Hidden/750',  # PA 750 data path
    'pa_850': './Data/Path_Hidden/850',  # PA 850 data path
    'us_750': './Data/Path_Hidden/750',  # US 750 data path
    'us_850': './Data/Path_Hidden/850'   # US 850 data path
}

best_hyperparameters = {
    'pa_750': {'C': 0.06690421166498801, 'gamma': 0.012561043700013555, 'kernel': 'linear'},
    'pa_850': {'C': 339.8172415010595, 'gamma': 0.00022592797420156976, 'kernel': 'sigmoid'},
    'us_750': {'C': 12.746711578215052, 'gamma': 0.005762487216478602, 'kernel': 'sigmoid'},
    'us_850': {'C': 12.746711578215052, 'gamma': 0.005762487216478602, 'kernel': 'sigmoid'}
}

# select dataset from above
dataset_name = 'pa_750'
dataset_path = dataset_paths[dataset_name]
hyperparameters = best_hyperparameters[dataset_name]

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

# use k-fold cross validation for grouped datasets (k-1/k train, 1/k val)
n_splits = 4
gkf = GroupKFold(n_splits=n_splits)

train_datagen = ImageDataGenerator(rescale=1.0/255)
val_datagen = ImageDataGenerator(rescale=1.0/255)

# Track fold performance
fold_accuracies = []

resnet50v2 = tf.keras.applications.ResNet50V2(input_shape=(512, 512, 3),
                                              include_top=False,
                                              pooling='avg',
                                              weights='imagenet')
model = resnet50v2
confusion_matrices = []

for fold, (train_idx, val_idx) in enumerate(gkf.split(df, df['class'], groups=df['mouse_id'])):
    print(f"\nStarting fold {fold + 1}/{n_splits}")

    # Split data for this fold
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

    # Create data generators
    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        x_col='filepath',
        y_col='class',
        target_size=(512, 512),
        batch_size=32,
        class_mode='sparse',
        shuffle=False
    )
    val_generator = val_datagen.flow_from_dataframe(
        val_df,
        x_col='filepath',
        y_col='class',
        target_size=(512, 512),
        batch_size=32,
        class_mode='sparse',
        shuffle=False
    )

    train_features, train_labels = extract_features(train_generator, model)
    val_features, val_labels = extract_features(val_generator, model)

    # normalize features because SVM classifiers are sensitive to the scale of features
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    val_features_scaled = scaler.transform(val_features)

    # train the classifier using tuned hyperparameters
    svm_clf = SVC(C=hyperparameters['C'], gamma=hyperparameters['gamma'], kernel=hyperparameters['kernel'], probability=True)
    svm_clf.fit(train_features_scaled, train_labels)

    val_predictions = svm_clf.predict(val_features_scaled)
    accuracy = accuracy_score(val_labels, val_predictions)
    fold_accuracies.append(accuracy)

    cm = confusion_matrix(val_labels, val_predictions)
    confusion_matrices.append(cm)

    print(f"Fold {fold + 1} Accuracy: {accuracy:.4f}")
    print(classification_report(val_labels, val_predictions))

# Print average validation accuracy across folds
average_accuracy = np.mean(fold_accuracies)
print(f"\nAverage validation accuracy across {n_splits} folds: {average_accuracy:.2f}")

# plotting confusion matrices
class_names = ['Severe', 'Sham', 'Mild']  # Example classes
titles = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4']
plot_multiple_confusion_matrices(
    conf_matrices=confusion_matrices,
    class_names=class_names,
    titles=titles,
    figsize=(10, 10)
)

