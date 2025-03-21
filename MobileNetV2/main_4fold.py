import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, log_loss
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from imblearn.over_sampling import RandomOverSampler

# --- Class Names ---
# Define the class labels for classification.
class_names = ['Mild', 'Sham', 'Severe']

# --- Data Preparation ---
# Function to prepare the dataset by reading image paths and labels.
def prepare_data():
    data, groups = [], []
    # Loop through each class and its associated folder.
    class_names = [
    'Class_1',  # Mild class
    'Class_2',  # Sham class
    'Class_3'   # Severe class
    ]

    for class_label, class_name in enumerate(class_names):
        # Define base directories for different imaging modalities and wavelengths.
        base_dirs = {
            'pa_750': './Data/Path_Hidden/750',  # PA 750 data path
            'pa_850': './Data/Path_Hidden/850',  # PA 850 data path
            'us_750': './Data/Path_Hidden/750',  # US 750 data path
            'us_850': './Data/Path_Hidden/850'   # US 850 data path
        }
        # Get the directory for the current class under a specific modality.
        class_dir = os.path.join(base_dirs['pa_750'], class_name)
        # Traverse through each mouse folder and collect image file paths.
        for mouse_id in os.listdir(class_dir):
            mouse_dir = os.path.join(class_dir, mouse_id)
            if os.path.isdir(mouse_dir):
                for filename in os.listdir(mouse_dir):
                    if filename.endswith(('.jpg', '.png')):
                        filepath = os.path.join(mouse_dir, filename)
                        data.append({'path': filepath, 'label': class_label})
                        groups.append(mouse_id)
    # Create a DataFrame from the collected data.
    df = pd.DataFrame(data)
    df['group'] = groups

    # Validate dataset integrity.
    assert not df.empty, "DataFrame is empty! Ensure all paths and files are valid."
    assert df['label'].nunique() > 1, "Insufficient number of labels for classification!"
    # Shuffle the dataset and reset index.
    return df.sample(frac=1, random_state=42).reset_index(drop=True)

# Load the prepared dataset.
df = prepare_data()

# --- Debugging Group Distribution ---
# Print the distribution of groups and labels for debugging purposes.
group_counts = df.groupby(['group', 'label']).size()
print("Group distribution per label:\n", group_counts)

# --- Data Augmentation ---
# Define a data augmentation pipeline using Keras layers.
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),  # Randomly flip images.
    tf.keras.layers.RandomRotation(0.2),                   # Randomly rotate images.
    tf.keras.layers.RandomZoom(0.2),                       # Randomly zoom in/out on images.
    tf.keras.layers.RandomBrightness(0.1),                 # Adjust brightness randomly.
    tf.keras.layers.RandomContrast(0.2)                    # Adjust contrast randomly.
])

# Function to apply data augmentation to a single image.
def augment_image(img_array):
    return data_augmentation(np.expand_dims(img_array, axis=0))[0].numpy()

# --- Feature Extraction ---
# Preprocessing layer for MobileNetV2.
preprocess_layer = tf.keras.applications.mobilenet_v2.preprocess_input

# Load the pre-trained MobileNetV2 model without the top classification layer.
base_model = tf.keras.applications.MobileNetV2(input_shape=(512, 512, 3), include_top=False, weights='imagenet')

# Define a feature extractor model by adding preprocessing and pooling layers.
feature_extractor = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(512, 512, 3)), # Input layer for image data.
    tf.keras.layers.Lambda(preprocess_layer),              # Apply MobileNetV2 preprocessing.
    base_model,                                            # Base MobileNetV2 model.
    tf.keras.layers.Dropout(0.5),                          # 50% dropout rate
    GlobalAveragePooling2D()                               # Pool features to a single vector.
])
base_model.trainable = False  # Freeze the base model weights.

# Function to extract features and labels for a given dataset split.
def extract_features_for_fold(df_subset, feature_extractor, augment=False):
    features, labels = [], []
    # Iterate through each row in the dataset.
    for _, row in tqdm(df_subset.iterrows(), total=df_subset.shape[0], desc="Extracting Features"):
        # Load the image and resize to the input size.
        img = tf.keras.preprocessing.image.load_img(row['path'], target_size=(512, 512))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        if augment:
            img_array = augment_image(img_array)  # Apply augmentation if specified.
        # Extract features using the feature extractor.
        features.append(feature_extractor.predict(np.expand_dims(img_array, axis=0))[0])
        labels.append(row['label'])
    return np.array(features), np.array(labels)

# --- Oversampling Function (Regular Oversampling) ---
# Function to apply regular oversampling for class balancing.
def oversample_data(features, labels):
    """
    Apply random oversampling to balance the dataset.
    
    Args:
        features (np.array): Feature vectors.
        labels (np.array): Corresponding labels.
    
    Returns:
        features_resampled, labels_resampled: Oversampled features and labels.
    """
    oversampler = RandomOverSampler(random_state=42)
    features_resampled, labels_resampled = oversampler.fit_resample(features, labels)
    return features_resampled, labels_resampled

# --- Custom Group K-Fold ---
def custom_group_kfold(df, n_splits):
    """
    Custom GroupKFold-like generator that ensures one mouse from each group is in the test set.
    
    Args:
        df (pd.DataFrame): The DataFrame containing 'label' and 'group' columns.
        n_splits (int): Number of folds.
    
    Yields:
        train_indices, test_indices: Indices for the train and test split.
    """
    unique_groups = df['group'].unique()
    labels = df.groupby('group')['label'].first().to_dict()  # Map group to their labels
    group_by_label = {label: [] for label in df['label'].unique()}
    
    # Organize groups by label
    for group in unique_groups:
        group_by_label[labels[group]].append(group)
    
    # Shuffle groups within each label
    for label in group_by_label:
        np.random.shuffle(group_by_label[label])
    
    # Create test groups for each fold
    test_folds = []
    for fold in range(n_splits):
        test_folds.append([
            group_by_label[label][fold % len(group_by_label[label])]
            for label in group_by_label
        ])
    
    # Generate train/test splits
    for test_groups in test_folds:
        test_indices = df[df['group'].isin(test_groups)].index
        train_indices = df[~df['group'].isin(test_groups)].index
        yield train_indices, test_indices

# --- Group K-Fold Training ---
history_per_fold = []
n_splits = 4  # Number of folds
custom_folds = custom_group_kfold(df, n_splits)

for fold_idx, (train_indices, val_indices) in enumerate(custom_folds):
    train_df = df.iloc[train_indices]
    val_df = df.iloc[val_indices]

    print(f"--- Fold {fold_idx + 1} ---")
    print(f"Training groups: {train_df['group'].unique()}")
    print(f"Validation groups: {val_df['group'].unique()}")

    X_train, y_train = extract_features_for_fold(train_df, feature_extractor, augment=True)
    X_val, y_val = extract_features_for_fold(val_df, feature_extractor, augment=False)

    X_train, y_train = oversample_data(X_train, y_train)

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

    # SVM Classifier
    # SVM Classifier with class_weight passed during initialization
    svm_pipeline = make_pipeline(
        StandardScaler(), 
        SVC(probability=True, class_weight=class_weight_dict)  # Pass class_weight directly here
    )

    # Grid search with the specified hyperparameters
    param_grid = {'svc__C': [0.1, 1, 10, 100], 'svc__kernel': ['linear', 'rbf', 'poly']}
    svm_pipeline = make_pipeline(
    StandardScaler(),
    SVC(probability=True, class_weight=class_weight_dict)
    )
    grid = GridSearchCV(svm_pipeline, param_grid, cv=3, n_jobs=-1)

    # Fit the model (no need to pass class_weight here)
    grid.fit(X_train, y_train)

    best_params = grid.best_params_
    print("Best Hyperparameters for the dataset:", best_params)

    best_svm = grid.best_estimator_

    train_acc = best_svm.score(X_train, y_train)
    val_acc = best_svm.score(X_val, y_val)


    # Predict probabilities instead of class labels
    y_pred_proba = best_svm.predict_proba(X_val)
    y_pred = best_svm.predict(X_val)
    x_pred = best_svm.predict(X_train)
    x_pred_proba = best_svm.predict_proba(X_train)

    # Calculate metrics
    val_loss = log_loss(y_val, y_pred_proba, labels=[0, 1, 2])
    train_loss = log_loss(y_train, x_pred_proba, labels=[0, 1, 2])
    train_acc = best_svm.score(X_train, y_train)
    val_acc = best_svm.score(X_val, y_val)

    # Append all metrics to history_per_fold once
    history_per_fold.append({
        'train_acc': train_acc,
        'val_acc': val_acc,
        'train_loss': train_loss,
        'val_loss': val_loss
    })

    print(classification_report(y_val, y_pred))
    cm = confusion_matrix(y_val, y_pred, labels=[0, 1, 2])
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names).plot(cmap='Blues')
    plt.title(f"Fold {fold_idx + 1} Confusion Matrix")
    plt.show()

# --- Final Plotting ---
def plot_fold_performance(history):
    train_accs = [fold['train_acc'] for fold in history]
    val_accs = [fold['val_acc'] for fold in history]
    train_losses = [fold['train_loss'] for fold in history]
    val_losses = [fold['val_loss'] for fold in history]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.set_xlabel('Fold')
    ax1.set_ylabel('Accuracy', color='tab:blue')
    ax1.plot(train_accs, label='Train Accuracy', color='tab:blue', marker='o')
    ax1.plot(val_accs, label='Validation Accuracy', color='tab:orange', marker='o')
    ax1.legend(loc='upper left')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax2 = ax1.twinx()  # Create a secondary axis for loss
    ax2.set_ylabel('Validation Loss', color='tab:red')
    ax2.plot(train_losses, label='Training Loss', color='tab:green', linestyle='--', marker='x')
    ax2.plot(val_losses, label='Validation Loss', color='tab:red', linestyle='--', marker='x')
    ax2.legend(loc='upper right')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    fig.tight_layout()
    plt.title('Accuracy and Loss Across Folds')
    plt.show()

plot_fold_performance(history_per_fold)
