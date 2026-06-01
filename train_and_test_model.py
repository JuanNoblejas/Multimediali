import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier

data_dir = 'data'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'validation')
test_dir = os.path.join(data_dir, 'test')
img_width, img_height = 150, 150
batch_size = 32
epochs = 10

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(img_width, img_height),
    batch_size=batch_size, class_mode='binary', shuffle=False)
val_generator = val_datagen.flow_from_directory(val_dir, target_size=(img_width, img_height),
    batch_size=batch_size, class_mode='binary')
test_generator = val_datagen.flow_from_directory(test_dir, target_size=(img_width, img_height),
    batch_size=1, class_mode='binary', shuffle=False)

if os.path.exists('modelo_neumonia.h5'):
    print("Loading existing CNN model...")
    model = load_model('modelo_neumonia.h5')
else:
    print("Training CNN model...")
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(img_width, img_height, 3)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_generator, steps_per_epoch=train_generator.samples // batch_size,
        validation_data=val_generator, validation_steps=val_generator.samples // batch_size,
        epochs=epochs)
    model.save('modelo_neumonia.h5')

predictions = model.predict(test_generator)
predicted_classes = (predictions > 0.5).astype(int).flatten()

mobilenet = MobileNetV2(input_shape=(img_width, img_height, 3), include_top=False,
                        weights='imagenet', pooling='avg')
mobilenet.save('modelo_mobilenet_features.h5')

train_features = mobilenet.predict(train_generator)
test_features = mobilenet.predict(test_generator)

if os.path.exists('modelo_rf.pkl'):
    print("Loading existing Random Forest model...")
    rf = joblib.load('modelo_rf.pkl')
else:
    print("Training Random Forest model...")
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf.fit(train_features, train_generator.classes)
    joblib.dump(rf, 'modelo_rf.pkl')

y_pred_rf = rf.predict(test_features)

if os.path.exists('modelo_mobilenet_finetuned.h5'):
    print("Loading existing fine-tuned MobileNetV2 model...")
    model_ft = load_model('modelo_mobilenet_finetuned.h5')
else:
    print("Training MobileNetV2 fine-tuned model (phase 1)...")
    base = MobileNetV2(input_shape=(img_width, img_height, 3), include_top=False, weights='imagenet')
    base.trainable = False

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)

    model_ft = Model(inputs=base.input, outputs=output)
    model_ft.compile(optimizer=Adam(0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    model_ft.fit(train_generator, steps_per_epoch=train_generator.samples // batch_size,
        validation_data=val_generator, validation_steps=val_generator.samples // batch_size,
        epochs=epochs)

    print("Fine-tuning phase 2 (last 20 layers)...")
    base.trainable = True
    for layer in base.layers[:-20]:
        layer.trainable = False
    model_ft.compile(optimizer=Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    model_ft.fit(train_generator, steps_per_epoch=train_generator.samples // batch_size,
        validation_data=val_generator, validation_steps=val_generator.samples // batch_size,
        epochs=5)
    model_ft.save('modelo_mobilenet_finetuned.h5')

preds_ft = model_ft.predict(test_generator)
y_pred_ft = (preds_ft > 0.5).astype(int).flatten()
y_true = test_generator.classes

def compute_metrics(y_true, y_pred):
    return {
        "accuracy":  round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1":        round(float(f1_score(y_true, y_pred, zero_division=0)), 4)
    }

metrics = {
    "CNN":                    compute_metrics(y_true, predicted_classes),
    "MobileNetV2 + RF":       compute_metrics(y_true, y_pred_rf),
    "MobileNetV2 Fine-tuned": compute_metrics(y_true, y_pred_ft)
}

with open('model_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("\n=== Comparative Metrics ===")
for name, m in metrics.items():
    print(f"\n{name}:")
    print(f"  Accuracy:  {m['accuracy']:.2%}")
    print(f"  Precision: {m['precision']:.2%}")
    print(f"  Recall:    {m['recall']:.2%}")
    print(f"  F1-Score:  {m['f1']:.2%}")

print("\n=== Detailed Reports ===")
print("--- CNN ---")
print(classification_report(y_true, predicted_classes, target_names=['Normal', 'Pneumonia']))
print("--- MobileNetV2 + RF ---")
print(classification_report(y_true, y_pred_rf, target_names=['Normal', 'Pneumonia']))
print("--- MobileNetV2 Fine-tuned ---")
print(classification_report(y_true, y_pred_ft, target_names=['Normal', 'Pneumonia']))

def plot_confusion(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Pneumonia'],
                yticklabels=['Normal', 'Pneumonia'])
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

plot_confusion(y_true, predicted_classes, "Confusion Matrix - CNN")
plot_confusion(y_true, y_pred_rf,         "Confusion Matrix - MobileNetV2 + RF")
plot_confusion(y_true, y_pred_ft,         "Confusion Matrix - MobileNetV2 Fine-tuned")
