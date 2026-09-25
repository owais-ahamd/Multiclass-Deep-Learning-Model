from io import StringIO
from math import ceil
from pathlib import Path
import tkinter as tk
from tkinter import ttk

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Activation, Dense, Dropout, Embedding, GlobalAveragePooling1D, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


# Load comments and their multiclass toxicity labels.
csv_path = Path(__file__).resolve().parent / "toxic_comments_dataset.csv"
df = pd.read_csv(csv_path)
X_text = df["Comment_Text"].fillna("").astype(str).str.lower().to_numpy()
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["Toxicity_Label"].fillna("Unknown").astype(str))

# Split before fitting the tokenizer to prevent information from test comments
# from entering the training vocabulary.
X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text, y, test_size=0.20, random_state=42, stratify=y
)
max_vocab = 5000
tokenizer = Tokenizer(num_words=max_vocab, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train_text)
X_train_sequences = tokenizer.texts_to_sequences(X_train_text)
X_test_sequences = tokenizer.texts_to_sequences(X_test_text)

# The 95th percentile of training sequence lengths keeps most comments intact.
train_lengths = [len(sequence) for sequence in X_train_sequences]
max_sequence_length = max(1, ceil(float(np.percentile(train_lengths, 95))))
X_train_padded = pad_sequences(
    X_train_sequences, maxlen=max_sequence_length, padding="post", truncating="post"
)
X_test_padded = pad_sequences(
    X_test_sequences, maxlen=max_sequence_length, padding="post", truncating="post"
)

# Multiclass neural network: Embedding -> pooling -> Dense -> Dropout -> Dense -> Softmax.
vocabulary_size = min(max_vocab, len(tokenizer.word_index) + 1)
num_classes = len(label_encoder.classes_)
model = Sequential([
    Input(shape=(max_sequence_length,)),
    Embedding(input_dim=vocabulary_size, output_dim=64),
    GlobalAveragePooling1D(),
    Dense(64, activation="relu"),
    Dropout(0.40),
    Dense(num_classes),
    Activation("softmax"),
])
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

epochs = 15
batch_size = 64
history = model.fit(
    X_train_padded,
    y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=0.20,
    verbose=2,
)
test_loss, test_accuracy = model.evaluate(X_test_padded, y_test, verbose=0)

model_summary = StringIO()
model.summary(print_fn=lambda line: model_summary.write(line + "\n"))
history_table = pd.DataFrame({
    "Epoch": range(1, len(history.history["loss"]) + 1),
    "Training loss": history.history["loss"],
    "Training accuracy": history.history["accuracy"],
    "Validation loss": history.history["val_loss"],
    "Validation accuracy": history.history["val_accuracy"],
})

print("Task 4: Multiclass Deep Learning Model")
print(f"Training records: {len(X_train_text):,}; testing records: {len(X_test_text):,}")
print(f"Classes: {', '.join(label_encoder.classes_)}")
print(f"Maximum sequence length: {max_sequence_length}")
print(f"Model trained for {len(history.history['loss'])} epochs; batch size: {batch_size}")
print(f"Test loss: {test_loss:.4f}; test accuracy: {test_accuracy:.4f}")
print("\nModel architecture:\n" + model_summary.getvalue())
print("Training history:\n" + history_table.round(4).to_string(index=False))


def show_dataframe(parent, frame):
    """Display a DataFrame in a scrollable table."""
    table_frame = ttk.Frame(parent)
    table_frame.pack(fill="both", expand=True, padx=10, pady=10)
    columns = list(frame.columns)
    table = ttk.Treeview(table_frame, columns=columns, show="headings")
    vertical_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=table.yview)
    horizontal_scroll = ttk.Scrollbar(table_frame, orient="horizontal", command=table.xview)
    table.configure(yscrollcommand=vertical_scroll.set, xscrollcommand=horizontal_scroll.set)
    for column in columns:
        table.heading(column, text=column)
        table.column(column, width=150, minwidth=100, stretch=True)
    for row in frame.itertuples(index=False, name=None):
        table.insert("", "end", values=row)
    table.grid(row=0, column=0, sticky="nsew")
    vertical_scroll.grid(row=0, column=1, sticky="ns")
    horizontal_scroll.grid(row=1, column=0, sticky="ew")
    table_frame.rowconfigure(0, weight=1)
    table_frame.columnconfigure(0, weight=1)


root = tk.Tk()
root.title("Task 4: Multiclass Deep Learning Model")
root.geometry("1200x700")
notebook = ttk.Notebook(root)
notebook.pack(fill="both", expand=True)

overview_tab = ttk.Frame(notebook)
notebook.add(overview_tab, text="Model Overview")
overview = (
    "Multi-class comment classification with TensorFlow/Keras\n\n"
    f"Classes: {', '.join(label_encoder.classes_)}\n"
    f"Training records: {len(X_train_text):,}\n"
    f"Testing records: {len(X_test_text):,}\n"
    f"Training validation split: 20% of training data\n"
    f"Epochs: {len(history.history['loss'])}; batch size: {batch_size}\n"
    f"Maximum sequence length: {max_sequence_length}\n"
    f"Test loss: {test_loss:.4f}\n"
    f"Test accuracy: {test_accuracy:.4f}\n\n"
    "Model architecture:\n"
    f"{model_summary.getvalue()}"
)
overview_text = tk.Text(overview_tab, wrap="none", font=("Consolas", 10))
overview_text.insert("1.0", overview)
overview_text.configure(state="disabled")
overview_text.pack(fill="both", expand=True, padx=10, pady=10)

history_tab = ttk.Frame(notebook)
notebook.add(history_tab, text="Training History")
show_dataframe(history_tab, history_table.round(4))

evaluation_tab = ttk.Frame(notebook)
notebook.add(evaluation_tab, text="Test Evaluation")
predictions = model.predict(X_test_padded[:20], verbose=0)
evaluation_table = pd.DataFrame({
    "Actual label": label_encoder.inverse_transform(y_test[:20]),
    "Predicted label": label_encoder.inverse_transform(np.argmax(predictions, axis=1)),
    "Confidence": np.max(predictions, axis=1).round(4),
})
show_dataframe(evaluation_tab, evaluation_table)

root.mainloop()
