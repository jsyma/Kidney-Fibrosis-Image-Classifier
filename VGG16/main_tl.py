import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

def get_available_devices():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]

# --- Enable GPU --- 
for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

# --- Base Directory of Project ---
base_dir = os.path.dirname(os.path.abspath(__file__))

# --- Note: make sure not to use the same mouse for testing and validation ---

# --- Paths to Training and Testing datasets ---
training_datasets = {
    "pa_750": os.path.join(base_dir, "Path_Hidden", "750", "training"),
    "pa_850": os.path.join(base_dir, "Path_Hidden", "850", "training"),
    "us_750": os.path.join(base_dir, "Path_Hidden", "750", "training"),
    "us_850": os.path.join(base_dir, "Path_Hidden", "850", "training"),
}

testing_datasets = {
    "pa_750": os.path.join(base_dir, "Path_Hidden", "750", "testing"),
    "pa_850": os.path.join(base_dir, "Path_Hidden", "850", "testing"),
    "us_750": os.path.join(base_dir, "Path_Hidden", "750", "testing"),
    "us_850": os.path.join(base_dir, "Path_Hidden", "850", "testing"),
}

# --- Select dataset path ---
dataset = "pa_750"
training_path = training_datasets[dataset]
testing_path = testing_datasets[dataset]

print(f"Using Training path: {training_path} and Testing path: {testing_path}")

# --- Load datasets ---
training_dataset = tf.keras.utils.image_dataset_from_directory(
    training_path,
    labels='inferred',
    image_size=(224, 224),
    batch_size=16,
    seed=123
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    testing_path,
    labels='inferred',
    image_size=(224, 224),
    batch_size=16,
    seed=123
)

# --- Normalize pixel values ---
normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
training_dataset = training_dataset.map(lambda x, y: (normalization_layer(x), y))
validation_dataset = validation_dataset.map(lambda x, y: (normalization_layer(x), y))

# --- Class weights to handle imbalance ---
class_names = ['Severe', 'Sham', 'Mild']
class_counts = [len(list(training_dataset.unbatch().filter(lambda x, y: y == i))) for i in range(len(class_names))]
total_samples = sum(class_counts)
class_weights = {i: total_samples / (len(class_names) * class_counts[i]) for i in range(len(class_names))}

# --- Use VGG16 as the base model ---
base_model = VGG16(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

# --- Fine-tune last few layers of VGG16 ---
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
    layers.GlobalAveragePooling2D(),
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

# --- Early stopping ---
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# --- Train the model ---
history = model.fit(
    training_dataset,
    epochs=100,
    validation_data=validation_dataset,
    class_weight=class_weights,
    callbacks=[early_stopping]
)

# --- Evaluate the model ---
val_labels = []
val_predictions = []

for image_batch, label_batch in validation_dataset:
    predictions = model.predict(image_batch)
    predicted_labels = tf.argmax(predictions, axis=1).numpy()
    val_predictions.extend(predicted_labels)
    val_labels.extend(label_batch.numpy())

# --- Confusion Matrix Plot---
cm = confusion_matrix(val_labels, val_predictions, labels=list(range(len(class_names))))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix')
plt.show()

# --- Classification report ---
report = classification_report(val_labels, val_predictions, target_names=class_names)
print("Classification Report:\n", report)

# --- Accuracy and Loss Plots ---
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Loss')
plt.show()
