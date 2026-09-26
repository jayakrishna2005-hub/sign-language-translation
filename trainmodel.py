import json
import os

import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from function import DATA_PATH, actions, sequence_length


def numeric_dirs(path):
    if not os.path.isdir(path):
        return []
    return sorted(
        [name for name in os.listdir(path) if name.isdigit()],
        key=lambda value: int(value),
    )


def load_dataset():
    label_map = {label: num for num, label in enumerate(actions)}
    sequences, labels = [], []
    skipped = []

    for action in actions:
        action_path = os.path.join(DATA_PATH, action)
        for sequence in numeric_dirs(action_path):
            window = []
            complete = True

            for frame_num in range(sequence_length):
                frame_path = os.path.join(action_path, sequence, f"{frame_num}.npy")
                if not os.path.exists(frame_path):
                    complete = False
                    break
                window.append(np.load(frame_path))

            if not complete:
                skipped.append((action, sequence, "missing frame"))
                continue

            window = np.array(window)
            if not np.any(window):
                skipped.append((action, sequence, "no hand landmarks"))
                continue

            sequences.append(window)
            labels.append(label_map[action])

    if not sequences:
        raise RuntimeError("No valid training sequences found in MP_Data.")

    return np.array(sequences), to_categorical(labels, num_classes=len(actions)).astype(int), skipped


def build_model():
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation="relu", input_shape=(30, 63)))
    model.add(Dropout(0.2))
    model.add(LSTM(128, return_sequences=True, activation="relu"))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=False, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(actions.shape[0], activation="softmax"))
    model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"])
    return model


X, y, skipped = load_dataset()

label_ids = np.argmax(y, axis=1)
stratify = label_ids if min(np.bincount(label_ids)) >= 2 else None
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.15,
    random_state=42,
    shuffle=True,
    stratify=stratify,
)

print(f"Loaded {len(X)} valid sequences.")
if skipped:
    print(f"Skipped {len(skipped)} invalid sequences.")

with open("label_map.json", "w") as file:
    json.dump({label: int(index) for index, label in enumerate(actions)}, file, indent=2)

model = build_model()
callbacks = [
    TensorBoard(log_dir=os.path.join("Logs")),
    ModelCheckpoint("best_model.weights.h5", monitor="val_categorical_accuracy", save_best_only=True, save_weights_only=True),
    EarlyStopping(monitor="val_categorical_accuracy", patience=30, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-5),
]

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=300,
    callbacks=callbacks,
    batch_size=16,
)

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Validation accuracy: {accuracy * 100:.2f}%")

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save("model.h5")
model.save("model_full.keras")

with open("training_history.json", "w") as file:
    json.dump(history.history, file, indent=2)

model.summary()
