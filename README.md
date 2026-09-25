# Task 4: Multiclass Deep Learning Model

`task4.py` trains a neural network to classify toxic comments into the categories in `Toxicity_Label`. It reads `toxic_comments_dataset.csv`, prepares text as padded integer sequences, trains and evaluates a TensorFlow/Keras model, and displays the model summary, training history, and sample test predictions in a Tkinter window.

## Workflow

- Uses `Comment_Text` as the input and `Toxicity_Label` as the target. Missing comments become empty strings; missing labels become `Unknown`.
- Encodes class labels as integers and makes a stratified 80/20 train/test split with random seed 42.
- Fits a tokenizer on training comments only, using a 5,000-word vocabulary limit and an `<OOV>` token.
- Pads or truncates comments to the 95th percentile of training sequence lengths.
- Trains an embedding-based multiclass neural network with global average pooling, a dense layer, dropout, and a softmax output layer.
- Trains for up to 15 epochs with batch size 64 and uses 20% of the training data for validation.
- Evaluates on the held-out test set and displays up to 20 actual and predicted labels with confidence scores.

## Requirements

- Python 3
- pandas
- NumPy
- scikit-learn
- TensorFlow
- Tkinter (usually included with desktop Python installations)

Install the packages with:

```bash
python -m pip install pandas numpy scikit-learn tensorflow
```

## Run

Keep `task4.py` and `toxic_comments_dataset.csv` in the same directory, then run:

```bash
python task4.py
```

Training runs each time the script starts. A desktop session is needed to open the Tkinter results window.

## Expected dataset columns

The CSV must contain `Comment_Text` and `Toxicity_Label`. The former supplies comment text; the latter supplies the classification categories.
