# Classification_Conv2net_MNIST
# Modèle CNN pour la Classification MNIST

Ce projet contient un exemple d'entraînement d'un modèle de réseau de neurones convolutifs (CNN) pour la classification des chiffres manuscrits du jeu de données MNIST en utilisant TensorFlow et Keras.

## Importation des bibliothèques nécessaires

```python
# Importation des bibliothèques nécessaires
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

## Chargement du jeu de données MNIST
# Charger le jeu de données MNIST
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

## Prétraitement des données
# Prétraitement des données
# Redimensionner les images et normaliser les valeurs des pixels
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

# Encodage one-hot des étiquettes
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

## Création du modèle CNN
# Création du modèle CNN
# Ajouter des couches convolutives et augmenter la complexité
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))  # Ajouter de la régularisation Dropout
model.add(layers.Dense(10, activation='softmax'))

## Compilation du modèle
# Compilation du modèle
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
## Entraînement du modèle
# Entraînement du modèle
history = model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.2)

## Évaluation du modèle
# Évaluation du modèle
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Accuracy sur le jeu de test : {test_acc}')

## Affichage de l'historique de l'entraînement
# Affichage de l'historique de l'entraînement
import matplotlib.pyplot as plt

accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(accuracy) + 1)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

## Récupération des images téléchargées
# Récupérer les images téléchargées
uploaded_images = list(uploaded.values())
