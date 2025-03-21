import os
import re
import numpy as np
import tensorflow as tf
from sklearn.model_selection import GroupKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, Sequential
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

tf.keras.backend.clear_session()
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# --- Enable GPU --- 
for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

# --- Base Directory of Project ---
base_dir = os.path.dirname(os.path.abspath(__file__))

# --- Paths to datasets ---
datasets = {
    "pa_750": os.path.join(base_dir, "Path_Hidden", "750"),
    "pa_850": os.path.join(base_dir, "Path_Hidden", "850"),
    "us_750": os.path.join(base_dir, "Path_Hidden", "750"),
    "us_850": os.path.join(base_dir, "Path_Hidden", "850"),
}

# --- Set Dataset Path ---
dataset = "us_850"
dataset_path = datasets[dataset]
print(f"Using dataset path: {dataset_path}")

# --- Severity class mapping ---
severity_map = {
    'Class_Hidden_1': 0,  # Severe
    'Class_Hidden_2': 1,  # Sham
    'Class_Hidden_3': 2   # Mild
}

image_paths = []
labels = []
group_labels = []

# --- Extract mouse ID from filenames and create dataset ---
severity_folders = severity_map.keys()
for severity in severity_folders:
    severity_path = os.path.join(dataset_path, severity)
    for filename in os.listdir(severity_path):
        if filename.endswith('.jpg'):
            match = re.match(r"PA|B_(LL|L|RR|R|B).+\.jpg", filename)
            if match:
                mouse_id = match.group(1)
                image_paths.append(os.path.join(severity_path, filename))
                labels.append(severity_map[severity])
                group_labels.append(mouse_id)

# --- DataFrame for GroupKFold ---
df = pd.DataFrame({
    'image_path': image_paths,
    'label': labels,
    'group': group_labels
})

# --- Data Preprocessing ---
def preprocess_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = image / 255.0
    return image, label

def prepare_dataset(image_paths, labels, batch_size=8, augment=False):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(preprocess_image)
    if augment:
        augmentation_layer = Sequential([
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.2),
            layers.RandomContrast(0.2),
            layers.RandomTranslation(height_factor=0.2, width_factor=0.2)
        ])
        dataset = dataset.map(lambda x, y: (augmentation_layer(x, training=True), y))
    dataset = dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# --- Feature Extraction using VGG16 convolutional layers ---
def extract_features(model, dataset):
    features, labels = [], []
    for images, batch_labels in dataset:
        conv_output = model.predict(images)
        conv_output = conv_output.reshape(conv_output.shape[0], -1)
        features.append(conv_output)
        labels.extend(batch_labels.numpy())
    return np.vstack(features), np.array(labels)

# --- Define the base model (VGG16) for feature extraction ---
vgg16_base = VGG16(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
vgg16_base.trainable = False

feature_extractor = Sequential([
    vgg16_base,
    layers.GlobalAveragePooling2D()
])

class_names = ["Severe", "Sham", "Mild"]

# --- Group K-Fold Cross-Validation ---
n_splits = 4
gkf = GroupKFold(n_splits=n_splits)
conf_matrices = []

for fold, (train_idx, val_idx) in enumerate(gkf.split(df['image_path'], df['label'], groups=df['group'])):
    print(f"Training fold {fold + 1}/{n_splits}...")
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    
    train_dataset = prepare_dataset(train_df['image_path'].values, train_df['label'].values, augment=True)
    val_dataset = prepare_dataset(val_df['image_path'].values, val_df['label'].values)
    
    X_train, y_train = extract_features(feature_extractor, train_dataset)
    X_val, y_val = extract_features(feature_extractor, val_dataset)
    
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    pca = PCA(n_components=50)
    X_train = pca.fit_transform(X_train)
    X_val = pca.transform(X_val)
    
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'gamma': ['scale', 'auto', 0.01, 0.1, 0.5]
    }
    svm = SVC(class_weight='balanced')
    grid_search = GridSearchCV(svm, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_svm = grid_search.best_estimator_
    y_pred = best_svm.predict(X_val)

    # Add classification report
    print(f"Classification Report for Fold {fold + 1}:")
    print(classification_report(y_val, y_pred, target_names=class_names))

    cm = confusion_matrix(y_val, y_pred, labels=[0, 1, 2])
    conf_matrices.append(cm)

# --- Plot Multiple Confusion Matrices ---
def plot_multiple_confusion_matrices(conf_matrices, class_names, titles, figsize=(12, 8)):
    n = len(conf_matrices)
    rows = int(np.ceil(n / 2))
    cols = 2 if n > 1 else 1
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if n > 1 else [axes]
    
    for i, cm in enumerate(conf_matrices):
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(ax=axes[i], cmap='Blues', colorbar=False)
        axes[i].set_title(titles[i])
    
    for j in range(len(conf_matrices), len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()

titles = [f"Fold {i + 1}" for i in range(n_splits)]

plot_multiple_confusion_matrices(conf_matrices, class_names, titles)
