# Employee-Attrition-DNN-Classifier
Employee Retention Prediction using DNN
<br> A Deep Neural Network Classifier is trained on employee records to predict if an employee is at risk of leaving
<p>
  <br>Employee Attrition Prediction Summary Report

<br><br>1. Purpose
This report analyzes employee attrition using a Deep Neural Network (DNN) classification model. The model aims to predict employee attrition risk based on various employee and company-related features.

<br><br>2. Methodology
<br>2.1 Data Preparation
<br>Data Loading and Merging: Employee data from separate 'train' and 'test' CSV files were combined for comprehensive analysis.
<br>Data Cleaning: Irrelevant features (Employee ID) were removed. Missing data points were handled, ensuring a clean dataset for model training.
<br>Feature Engineering:
<br>Categorical Encoding: Categorical features (e.g., Gender, Job Role) were converted to numerical values using encoding techniques to facilitate model processing.
<br>Data Scaling: Numerical features were standardized using StandardScaler to have zero mean and unit variance, preventing features with large magnitudes from disproportionately influencing the model.
<br>Feature Correlation Analysis: Pearson correlation was employed to visualize the relationship between each feature and the target variable (Attrition).
<br>2.2 Model Training and Evaluation
<br>Data Splitting: The dataset was randomly split into training (60%), evaluation (20%), and test (20%) sets to train, tune hyperparameters, and assess the model's generalization performance.
<br>Model Architecture: A DNN with multiple dense layers and ReLU activation functions was chosen for its ability to capture complex non-linear relationships within the data.
<br>Model Compilation: The model was compiled using the 'adam' optimizer and 'sparse_categorical_crossentropy' loss function, suitable for multi-class classification problems.
<br>Callbacks:
<br>Learning Rate Scheduler: A custom learning rate scheduler was implemented to dynamically adjust the learning rate during training, potentially improving convergence speed and performance.
<br>Early Stopping: This callback halted training when the model's performance on the validation set stopped improving, preventing overfitting.
<br>Model Training: The model was trained on the training data, with progress monitored for loss reduction.
<br>Performance Metrics: The model was evaluated using accuracy, precision, recall, and F1-score on both the training and evaluation datasets.
<br>Confusion Matrix: A confusion matrix was generated to visualize the model's predictions, highlighting true positives, true negatives, false positives, and false negatives.
<br>2.3 Prediction
<br>New Data Input: A sample input DataFrame reflecting a potential employee's information was created.
<br>Prediction: The trained model was used to predict the attrition risk for the new input data.
<br><br>3. Observations
<br>Feature Importance: Marital status, job level, and remote work appear to have the most significant influence on attrition.
<br>Model Performance: The DNN model achieved a reasonable performance on the training and evaluation datasets, with metrics (precision, recall, F1-score, accuracy) ranging from 0.75 to 0.8.
<br>False Positives: Approximately 14% of employees predicted to stay actually left, representing a significant area of concern.
<br><br>4. Findings
<br>The developed DNN model can predict employee attrition risk with moderate accuracy.
<br>The model highlights key features that strongly correlate with employee attrition, providing insights for HR strategies and interventions.
<br>Addressing the false-positive rate is crucial to minimize unexpected employee departures and associated costs.
<br><br>5. Improvements
<br>Model Optimization: Experiment with hyperparameter tuning (e.g., number of layers, neurons per layer, learning rate, activation functions) to enhance model performance.
F<br>eature Engineering: Explore additional features or engineered features that might better capture attrition drivers (e.g., interaction terms, time-based variables).
<br>Continuous Monitoring and Refinement: Continuously evaluate the model's performance on new data and retrain it periodically to adapt to evolving trends and maintain accuracy.
<br>By implementing these improvements, the model's predictive power can be further enhanced, providing more accurate and actionable insights for reducing employee attrition and improving overall workforce management.
</p>
<br>Link to notebook: https://www.kaggle.com/code/kgraghav/predict-attrition-dnn
