# Kidney Fibrosis Image Classifier

This project employs computer vision techniques to classify kidney fibrosis in mice using photoacoustic and ultrasound images. The goal is to utilize machine learning techniques to classify these images into different severity levels based on their features. 

## Models 
We utilize transfer learning with 3 pre-trained models:
- **MobileNetV2**: Known for its lightweight architecture and speed.
- **VGG-16**: Offers strong generalizability across applications.
- **ResNet50V2**: A deeper model that can capture complex features.

## Dataset 
The dataset consists of labeled images categorized into three classes: sever, mild and sham. Due to confidentiality agreements, the data and specific dataset paths are hidden.

## Machine Learning Techniques Used
- **Transfer Learning**: Leveraged pre-trained models to speed up training and improve performance on our dataset.
- **Fine-Tuning**: Adjusted specific layers of pre-trained models while keeping others fixed to better fit our data.
- **Feature Extraction**: Used pre-trained models to extract important features from images for classification.
- **CatBoost**: Utilized for its ability to efficiently handle categorical features and enhance model accuracy with minimal preprocessing.
- **Support Vector Machines (SVM)**: Implemented as a classifier on extracted features, allowing for effective separation of classes.
- **Hyperparameter Tuning**: Systematically adjusted key parameters (like regularization parameter C, kernel type, and kernel coefficient gamma) to improve model accuracy.
- **Data Augmentation**: Increased the dataset size by creating variations of existing images, helping the model to generalize better.
- **Group K-Fold Cross Validation**: Applied to ensure robust model evaluation and prevent data leakage during training and testing.
- **Early Stopping**: Stopped training when the model's performance on validation data stopped improving, preventing overfitting.
- **Synthetic Minority Over-sampling Technique (SMOTE)**: Addressed class imbalance by creating synthetic examples for underrepresented classes, enhancing model training.

## Technologies Used 
- **Python**: The programming language used for development.
- **TensorFlow**: Library used for building and training deep learning models.
- **Keras**: Used as a high-level API for building and training neural networks.
- **Scikit-learn**: A library for machine learning, data analysis and model evaluation.
- **Pandas**: Used for data manipulation and analysis. 
- **NumPy**: Used for handling numerical arrays, crucial for processing image features and labels. 
- **Matplotlib**: Used for visualizing data and model performance. 