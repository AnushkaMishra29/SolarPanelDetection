# 🔋 Solar Panel Condition Classifier

A deep learning-based solution for classifying the condition of solar panels using image data. This project uses a convolutional neural network (ResNet18) implemented in PyTorch to identify six common panel conditions, with deployment via a user-friendly Streamlit app.

---

## 📌 Project Objectives

* **Classification Task:** Detect and classify solar panel images into:

  * Clean
  * Dusty
  * Bird-Drop
  * Electrical-Damage
  * Physical-Damage
  * Snow-Covered
* **Use Cases:**

  * Automate defect identification from images
  * Provide actionable maintenance insights
  * Help solar plant operators optimize panel efficiency

---

## 🧠 Project Pipeline

### 1. **Data Preprocessing**

* Resized all images to `224x224`
* Applied image augmentations to improve generalization:

  * Random horizontal flip
  * Random rotation
  * Color jitter
* Normalized images with ImageNet statistics (mean & std)

### 2. **Handling Class Imbalance**

* Used `WeightedRandomSampler` to oversample minority classes during training

### 3. **Model Training**

* Used a **ResNet18** model with fine-tuning
* Modified final fully connected layer to output 6 classes
* Optimized using Adam optimizer & CrossEntropyLoss

### 4. **Evaluation Metrics**

* Accuracy, Precision, Recall, F1-Score
* Confusion matrix
* Achieved **\~82% accuracy** with balanced performance across classes

### 5. **Model Saving**

* Saved as `solar_panel_classifier.pth`

---

## 🚀 Streamlit Web App

### Features:

* Upload solar panel images (JPG/PNG)
* Real-time condition classification
* Maintenance recommendation based on prediction

### To Run the App:

```bash
streamlit run app.py
```

### Output:

* **Predicted Class** (e.g., "Dusty")
* **Recommendation** (e.g., "Recommend cleaning panel soon.")

---

## 🗂 File Structure

```
├── data/                         # Image dataset (organized in subfolders)
├── solar_panel_training.py       # Model training script
├── app.py                        # Streamlit app
├── solar_panel_classifier.pth    # Trained model weights
├── README.md                     # Project documentation
```

---

## 📦 Requirements

```bash
pip install torch torchvision streamlit scikit-learn matplotlib seaborn pillow
```

---

## ✅ Results Snapshot

| Class             | Precision | Recall | F1-Score |
| ----------------- | --------- | ------ | -------- |
| Bird-drop         | 0.78      | 0.81   | 0.80     |
| Clean             | 0.81      | 0.93   | 0.87     |
| Dusty             | 0.79      | 0.70   | 0.74     |
| Electrical-damage | 0.90      | 0.95   | 0.92     |
| Physical-Damage   | 0.70      | 0.54   | 0.61     |
| Snow-Covered      | 0.95      | 0.83   | 0.89     |
