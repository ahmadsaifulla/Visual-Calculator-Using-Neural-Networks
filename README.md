# ANN Visual Calculator

This project is a handwritten-symbol visual calculator built entirely with **ANN-based** classifiers. It lets a user draw or upload three symbols in sequence — **digit, operator, digit** — predicts each symbol, and then evaluates the resulting equation.

The system has three main parts:
- **Training pipelines** for digit and operator recognition
- **Flask inference API** that loads trained models and serves predictions
- **Browser UI** for drawing/uploading symbols and displaying predictions/results

## 1. System overview

### Goal
The application recognizes handwritten-style digits and arithmetic operators, then computes the final expression.

Example flow:
1. User enters `5`, `·`, `3`
2. Frontend normalizes each image to a 28×28 ANN-compatible format
3. Frontend sends each image to the Flask API
4. API predicts the digit/operator classes with confidence
5. Frontend sends the predicted triplet to the calculator endpoint
6. Backend returns the final equation and result

### Current constraints
- **ANN only**: the system uses dense neural networks with dropout.
- **No CNN layers** are used anywhere in the active system.
- Digits and operators are trained as **separate models**.
- The preprocessing contract is intentionally aligned between frontend and backend.

## 2. Project structure

Key files:

- `app.py` — Flask backend and inference API
- `templates/index.html` — main UI markup
- `static/script.js` — browser interaction logic and client-side preprocessing
- `static/style.css` — UI styling
- `train_digit_ann.py` — digit ANN training pipeline
- `train_operator_ann.py` — operator ANN training pipeline
- `symbols/` — dataset root
- `digit_model_ann.keras` — trained digit model
- `operator_model_ann.keras` — trained operator model
- `digit_class_mapping.json` — digit class index mapping
- `operator_class_mapping.json` — operator class index mapping
- `digit_training_history.png` — digit training plot
- `operator_training_history.png` — operator training plot
- `digit_confusion_matrix.png` — digit confusion matrix image
- `operator_confusion_matrix.png` — operator confusion matrix image

## 3. Dataset layout

The training scripts expect this dataset structure:

```text
symbols/
  train/
    0/
    1/
    2/
    ...
    9/
    dot/
    minus/
    plus/
    slash/
  test/
    0/
    1/
    2/
    ...
    9/
    dot/
    minus/
    plus/
    slash/
```

### Digit classes
The digit model uses only:
- `0 1 2 3 4 5 6 7 8 9`

### Operator classes
The operator model reads folders:
- `dot`
- `minus`
- `plus`
- `slash`

These are mapped to calculator symbols as:
- `dot -> ·`
- `minus -> -`
- `plus -> +`
- `slash -> ÷`

### Image requirements
Both training pipelines expect each image to be:
- grayscale-convertible
- exactly **28×28** pixels

If an image has a different size, training raises an error.

## 4. Training pipelines

There are two separate training scripts:
- `train_digit_ann.py`
- `train_operator_ann.py`

Both scripts:
- load images from `symbols/train` and `symbols/test`
- normalize pixel values to `[0, 1]`
- flatten images to vectors of length **784**
- train an ANN classifier
- evaluate on the test set
- save the trained model and output artifacts

---

### 4.1 Digit training pipeline

The digit pipeline is the more advanced of the two.

#### File
- `train_digit_ann.py`

#### Data processing
The digit trainer:
1. Loads only the digit folders `0..9`
2. Converts each 28×28 image into a float array
3. Normalizes by dividing by `255.0`
4. Flattens each sample to length `784`
5. Converts labels to one-hot vectors

#### Validation split
The script creates a validation split from the training data instead of using the test set as validation.

This is done so the test split remains a true evaluation set.

#### Class weighting
Digit training uses balanced class weights to reduce bias toward easier or more frequent classes.

#### ANN architecture
The digit model uses only:
- `Input`
- `Dense`
- `Dropout`

No convolution, pooling, batch normalization, or other CNN-oriented layers are part of the active system.

#### Retry strategy
The trainer tries multiple ANN configurations through `ARCHITECTURE_RETRIES`.

Current retry dimensions include:
- hidden layer sizes
- dropout amount
- L2 regularization
- learning rate
- batch size

This allows the script to search a small ANN-only hyperparameter space without changing the rest of the application.

#### Training callbacks
Each retry uses:
- `EarlyStopping`
- `ReduceLROnPlateau`
- `ModelCheckpoint`

#### Confusion-aware tuning
Digit tuning has an explicit focus on reducing confusion between **2** and **7**.

The script logs:
- `2 -> 7`
- `7 -> 2`

It also uses this confusion information when deciding which retry candidate is best, especially when accuracies are close.

#### Best-model selection
The digit trainer promotes candidates based on:
1. acceptable overall accuracy
2. lower `2↔7` confusion when results are near ties

This keeps tuning focused on the most visible handwritten ambiguity without ignoring total accuracy.

#### Digit artifacts produced
Running `train_digit_ann.py` writes:
- `digit_model_ann.keras`
- `digit_class_mapping.json`
- `digit_training_history.png`
- `digit_confusion_matrix.png`

---

### 4.2 Operator training pipeline

The operator pipeline is simpler.

#### File
- `train_operator_ann.py`

#### Data processing
The operator trainer:
1. Loads only the operator folders `dot`, `minus`, `plus`, `slash`
2. Converts each image to grayscale
3. Normalizes pixel values to `[0, 1]`
4. Flattens to length `784`
5. Converts labels to one-hot vectors

#### ANN architecture
The operator model is also ANN-only and uses:
- `Input`
- `Dense`
- `Dropout`

#### Retry strategy
The operator trainer tries several dense/dropout configurations and stops once test accuracy reaches the configured acceptable threshold.

#### Operator artifacts produced
Running `train_operator_ann.py` writes:
- `operator_model_ann.keras`
- `operator_class_mapping.json`
- `operator_training_history.png`
- `operator_confusion_matrix.png`

## 5. Model artifacts

The runtime backend depends on these files existing:
- `digit_model_ann.keras`
- `operator_model_ann.keras`
- `digit_class_mapping.json`
- `operator_class_mapping.json`

If model files are missing, the Flask app raises a startup error and instructs the user to train the ANN models first.

## 6. Backend inference system

### File
- `app.py`

### Responsibilities
The backend is responsible for:
- loading both trained ANN models once at startup
- decoding incoming base64 images
- applying backend normalization/preprocessing
- running inference
- returning structured JSON responses
- evaluating the predicted equation result

### Startup behavior
At import/startup time, `app.py` loads:
- digit ANN model
- operator ANN model
- digit mapping JSON
- operator mapping JSON

This means inference is ready immediately after the Flask app starts, but startup will fail if artifacts are missing.

## 7. Backend preprocessing pipeline

The main backend preprocessing function is `preprocess_for_ann(...)` in `app.py`.

It converts an incoming image into the exact ANN input format expected by the models.

### Steps
1. Convert image to grayscale
2. Compute the mean pixel value
3. If the image is dark-on-light inverted, invert it
4. Build an ink mask using a threshold
5. Find the tight bounding box around the ink
6. Crop to that bounding box
7. Resize the cropped symbol to fit inside a **20×20** area
8. Paste it centered into a **28×28** white image
9. Normalize to `[0, 1]`
10. Flatten to shape `(1, 784)`

### Why this matters
This centering and scaling makes the live inference path closer to the training distribution and helps reduce mistakes caused by symbol position and size variation.

## 8. API endpoints

### `GET /`
Returns the main calculator page.

### `POST /predict_digit`
Predicts one digit.

#### Request body
```json
{
  "image": "data:image/png;base64,..."
}
```

#### Success response
```json
{
  "digit": 5,
  "confidence": 0.97,
  "probabilities": [ ... ]
}
```

#### Error responses
- `400` if the symbol cannot be recognized or the payload is invalid
- `500` if inference fails unexpectedly

---

### `POST /predict_operator`
Predicts one operator.

#### Request body
```json
{
  "image": "data:image/png;base64,..."
}
```

#### Success response
```json
{
  "operator": "+",
  "confidence": 0.95,
  "probabilities": [ ... ]
}
```

#### Error responses
- `400` if the symbol cannot be recognized or the payload is invalid
- `500` if inference fails unexpectedly

---

### `POST /calculate`
Evaluates the recognized expression.

#### Request body
```json
{
  "d1": 5,
  "op": "·",
  "d2": 3
}
```

#### Success response
```json
{
  "equation": "5 · 3",
  "result": 15
}
```

#### Supported operations
- `+`
- `-`
- `·`
- `÷`

#### Special case
Division by zero returns:
```json
{
  "equation": "7 ÷ 0",
  "result": "Error: Cannot divide by zero"
}
```

#### Error responses
- `400` for missing or invalid inputs
- `500` for unexpected failures

## 9. Frontend UI system

### Files
- `templates/index.html`
- `static/script.js`
- `static/style.css`

### Layout
The UI has three input boxes:
1. left digit box
2. middle operator box
3. right digit box

Each box contains:
- mode selector (`draw` or `upload`)
- drawing canvas or upload area
- clear button
- prediction symbol area
- confidence text
- low-confidence warning area

Below the three boxes is a controls panel with:
- **Calculate** button
- **Reset All** button
- status line
- equation/result display

## 10. Browser-side interaction flow

### State management
`static/script.js` stores per-box state in a `Map` keyed by:
- `d1`
- `op`
- `d2`

Each entry tracks:
- current mode
- canvas/context
- uploaded image data
- whether the box has input
- current prediction
- UI elements for that box

### Draw mode
In draw mode, the user draws on a 280×280 canvas.

Canvas behavior includes:
- pointer-based input
- rounded brush
- smooth stroke interpolation using quadratic curves
- white background with black ink

### Upload mode
In upload mode, the user can:
- click to select a file
- drag and drop an image
- preview the uploaded image

### Reset behavior
- **Clear** resets one box
- **Reset All** resets the whole interface, clears predictions, and restores the default waiting message

## 11. Frontend preprocessing pipeline

The browser-side normalization function is `normalizeImage(...)` in `static/script.js`.

Its purpose is to align browser-generated inputs with backend ANN expectations.

### Steps
1. Draw the source image into a 280×280 working canvas
2. Convert the pixels to grayscale
3. Estimate average brightness
4. Invert the image if the symbol appears light-on-dark
5. Find the ink bounding box using a threshold
6. Crop logically to the tight symbol region
7. Scale the symbol to fit inside a **20×20** area
8. Center it inside a **28×28** white canvas
9. Export the result as a base64 PNG

### Important note
The frontend and backend both implement crop-and-center preprocessing. This is intentional so the previewed input and server-side model input stay aligned.

## 12. Prediction and calculation flow in the browser

When the user clicks **Calculate**:
1. The frontend validates that all three boxes have input
2. It normalizes the digit/operator/digit images
3. It sends three prediction requests in parallel:
   - `/predict_digit`
   - `/predict_operator`
   - `/predict_digit`
4. It updates the UI with the predicted symbol and confidence for each box
5. It sends the predicted triplet to `/calculate`
6. It renders the final equation and result

### Confidence UI
The frontend uses three confidence bands:
- **good**: above `0.9`
- **warn**: `0.7` to `0.9`
- **bad**: below `0.7`

If confidence is below `0.7`, the UI shows:
- `Low confidence — try drawing clearer`

## 13. Styling system

### File
- `static/style.css`

### Role
The stylesheet defines:
- dark gradient application background
- panel layout and spacing
- 3-column responsive grid
- canvas presentation
- drag/drop styling
- buttons and controls
- confidence color states
- equation display transitions

At smaller widths, the 3-column layout collapses into a single-column stack.

## 14. Setup

## Python environment
Use the project virtual environment if available.

Example setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If there is no `requirements.txt`, install the libraries used by the codebase:
- Flask
- TensorFlow
- NumPy
- Pillow
- matplotlib
- scikit-learn

## 15. Running the app

Start the Flask app with:

```bash
./.venv/bin/python app.py
```

By default, the app runs on:
- `http://127.0.0.1:5050`

Then open the page in your browser.

## 16. Training the models

### Train the digit model
```bash
./.venv/bin/python train_digit_ann.py
```

### Train the operator model
```bash
./.venv/bin/python train_operator_ann.py
```

After training, confirm that the model and mapping files exist before starting the app.

## 17. Generated outputs after training

### Digit training outputs
- `digit_model_ann.keras`
- `digit_class_mapping.json`
- `digit_training_history.png`
- `digit_confusion_matrix.png`

### Operator training outputs
- `operator_model_ann.keras`
- `operator_class_mapping.json`
- `operator_training_history.png`
- `operator_confusion_matrix.png`

## 18. End-to-end request flow

```text
User draw/upload
    -> browser preprocessing (28x28 centered symbol)
    -> /predict_digit or /predict_operator
    -> backend preprocessing
    -> ANN inference
    -> prediction + confidence returned
    -> /calculate with predicted d1/op/d2
    -> equation/result JSON
    -> UI display
```

## 19. Current limitations

- Handwritten ambiguity still exists for some symbols, especially certain digit shapes.
- The digit model has had a specific tuning focus on reducing `2↔7` confusion, but difficult handwriting can still be misclassified.
- The system depends on preprocessing consistency; changes to one side should usually be mirrored on the other.
- Because the models are ANN-only, they may be less robust than a CNN-based classifier on more diverse handwriting — but the project intentionally keeps the ANN-only constraint.

## 20. Troubleshooting

### App fails at startup with missing model files
Train the models first:

```bash
./.venv/bin/python train_digit_ann.py
./.venv/bin/python train_operator_ann.py
```

### Port already in use
If port `5050` is occupied, stop the conflicting process or run the app on a different port.

### Predictions are unexpectedly poor
Check:
- symbol is centered and clearly drawn
- image background/foreground polarity is correct
- frontend and backend preprocessing are still aligned
- the current model artifacts were produced by the latest training scripts

### Training crashes on image shape errors
Make sure all dataset images are exactly **28×28**.

## 21. Maintenance guidance

If you modify this project later:
- keep the system **ANN-only** unless project requirements explicitly change
- preserve API contracts unless you update both frontend and backend together
- keep preprocessing behavior aligned between `static/script.js` and `app.py`
- retrain models after meaningful preprocessing or dataset changes
- inspect confusion matrices, not just accuracy, when tuning recognition quality

## 22. Summary

This project is a complete ANN-based visual calculator that:
- recognizes handwritten digits and operators
- preprocesses symbols consistently in the browser and backend
- predicts symbols through Flask inference endpoints
- computes the final arithmetic result
- supports retraining and tuning through dedicated ANN training scripts

It is structured so training, inference, and UI behavior are separated cleanly, while still sharing the same 28×28 centered-symbol contract.
