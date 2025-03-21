import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import re
import pandas as pd
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import GroupKFold
from tensorflow.keras.callbacks import EarlyStopping

def get_available_devices():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]


def create_model(class_names):
    base_model = tf.keras.applications.ResNet50V2(input_shape=(512, 512, 3),
                                                  include_top=False,
                                                  pooling='avg',
                                                  weights='imagenet')

    # Fine-tune last few layers of base model
    base_model.trainable = True
    for layer in base_model.layers[:-4]:
        layer.trainable = False

    # --- Data augmentation ---
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.1),
    ])

    # --- Build the model ---
    model = tf.keras.Sequential([
        data_augmentation,
        base_model,
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(len(class_names), activation='softmax')
    ])

    # --- Compile the model ---
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


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


# --- Enable GPU --- 
for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

# --- Paths to datasets ---
dataset_paths = {
    'pa_750': './Data/Path_Hidden/750',  # PA 750 data path
    'pa_850': './Data/Path_Hidden/850',  # PA 850 data path
    'us_750': './Data/Path_Hidden/750',  # US 750 data path
    'us_850': './Data/Path_Hidden/850'   # US 850 data path
}

# --- Select dataset path ---
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

# use k-fold cross validation for grouped datasets (k-1/k train, 1/k val)
n_splits = 4
gkf = GroupKFold(n_splits=n_splits)

train_datagen = ImageDataGenerator(rescale=1.0/255)
val_datagen = ImageDataGenerator(rescale=1.0/255)

# Track fold performance
fold_accuracies = []
history_list = []
epochs = 100

# --- Early stopping ---
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

class_names = ['Severe', 'Sham', 'Mild']
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
        shuffle=True
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

    # --- Class weights to handle imbalance ---
    class_indices = train_generator.class_indices
    class_counts = [0, 0, 0]
    for i, row in df.iterrows():
        data_class = row['class']
        idx = class_indices[data_class]
        class_counts[idx] += 1

    total_samples = sum(class_counts)
    class_weights = {i: total_samples / (len(class_names) * class_counts[i]) for i in range(len(class_names))}
    print(f'class counts: {class_counts}')
    print(f'total samples: {total_samples}')
    print(f'class weights: {class_weights}')

    model = create_model(class_names)
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        class_weight=class_weights,
        callbacks=[early_stopping]
    )
    history_list.append(history)

    # Get validation accuracy for this fold
    fold_accuracies.append(history.history['val_accuracy'][-1])

    # --- Evaluate the model ---
    val_labels = val_generator.labels
    val_predictions = model.predict(val_generator, verbose=1)
    val_predictions_classes = np.argmax(val_predictions, axis=1)

    # --- Confusion Matrix Plot---
    cm = confusion_matrix(val_labels, val_predictions_classes)
    confusion_matrices.append(cm)

    # --- Classification report ---
    report = classification_report(val_labels, val_predictions_classes, target_names=class_names)
    print("Classification Report:\n", report)

    # --- Accuracy and Loss Plots ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))

    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', linestyle='--')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].set_title('Model Accuracy')

    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss', linestyle='--')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].set_title('Model Loss')

    plt.tight_layout()
    plt.show()

average_accuracy = np.mean(fold_accuracies)
print(f"\nAverage validation accuracy across {n_splits} folds: {average_accuracy:.2f}")

# plotting confusion matrices
titles = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4']
plot_multiple_confusion_matrices(
    conf_matrices=confusion_matrices,
    class_names=class_names,
    titles=titles,
    figsize=(10, 10)
)