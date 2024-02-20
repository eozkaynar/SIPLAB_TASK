# Pulse Transit Time Calculation
Preprocessing: Apply filters and denoising techniques to preprocess signals such as BCG, PPG, and ECG.

QRS Detection: Detect R peaks in the ECG signal using the Pan-Tompkins algorithm.

Feature Extraction: Extract features from signals such as PPG max-min points, BCG I-J-K waves, and BP max-min points.

PTT Calculation: Calculate Pulse Transit Time (PTT) from finger PPG and BCG signals only.

Prediction: Split the dataset into training and testing sets, create linear regression models for predicting systolic (SBP) and diastolic (DBP) blood pressure, fit the models, and make predictions.

Evaluation: Evaluate the performance of the models by calculating correlation coefficients between predicted and actual blood pressure values.

## Peak Detection
<img src="peak_detection.png" width="1000">
