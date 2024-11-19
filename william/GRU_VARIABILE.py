import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense, Masking, Layer
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import Counter

# Caricamento del dataset
data = pd.read_csv("dataset_completo.csv", index_col=0)

# Definizione delle colonne delle feature e della label
features = ['AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ']
label_col = 'label'

# Funzione per creare sequenze variabili
def create_variable_length_sequences(data, feature_columns, label_column):
    X, y = [], []
    current_sequence = []
    current_label = None

    for i, row in data.iterrows():
        if current_label is None:  # Prima riga
            current_label = row[label_column]

        if row[label_column] == current_label:  # Continua la stessa azione
            current_sequence.append(row[feature_columns].values)
        else:  # Cambia azione
            X.append(np.array(current_sequence))
            y.append(current_label)
            current_sequence = [row[feature_columns].values]
            current_label = row[label_column]

    # Aggiungi l'ultima sequenza
    if current_sequence:
        X.append(np.array(current_sequence))
        y.append(current_label)

    return X, y

# Creazione delle sequenze e delle etichette
X, y = create_variable_length_sequences(data, features, label_col)

# Conversione delle etichette in one-hot encoding
y = to_categorical(y, num_classes=5)

# Padding delle sequenze variabili
X_padded = pad_sequences(X, padding='post', dtype='float32')

# Verifica delle sequenze
print(f"Shape delle sequenze dopo padding: {X_padded.shape}")
print(f"Distribuzione delle classi: {Counter(np.argmax(y, axis=1))}")

# Suddivisione in training e test set
X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.2, random_state=42)

# Livello di attenzione personalizzato con gestione della maschera
class AttentionWithMasking(Layer):
    def __init__(self, **kwargs):
        super(AttentionWithMasking, self).__init__(**kwargs)
        self.score_dense = Dense(1, activation='tanh')  # Livello Dense creato nel costruttore

    def call(self, inputs, mask=None):
        hidden_states = inputs

        # Calcolo del punteggio di attenzione
        score = self.score_dense(hidden_states)
        score = tf.squeeze(score, axis=-1)  # Rimuove l'ultimo asse

        # Applica maschera se presente
        if mask is not None:
            mask = tf.cast(mask, dtype=score.dtype)
            score += (1.0 - mask) * -1e9  # Penalizza timestep mascherati

        # Calcolo dei pesi di attenzione
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = tf.reduce_sum(hidden_states * tf.expand_dims(attention_weights, -1), axis=1)

        return context_vector, attention_weights

    def compute_mask(self, inputs, mask=None):
        return None  # Non propaga la maschera

# Costruzione del modello
inputs = Input(shape=(None, len(features)))  # Sequenze variabili
masked_inputs = Masking(mask_value=0.0)(inputs)  # Aggiunge maschera per padding
gru_output = GRU(64, return_sequences=True)(masked_inputs)

# Livello di attenzione
attention_layer = AttentionWithMasking()
context_vector, attention_weights = attention_layer(gru_output)  # La maschera è gestita automaticamente

# Strati finali
dense_output = Dense(32, activation='relu')(context_vector)
outputs = Dense(5, activation='softmax')(dense_output)

# Creazione e compilazione del modello
model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Sommario del modello
model.summary()

# Addestramento del modello
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=64,
    verbose=1
)

# Valutazione sul test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Funzione per plottare i grafici
def plot_training_history(history):
    plt.figure(figsize=(12, 5))

    # Perdita
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Accuratezza
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_training_history(history)
