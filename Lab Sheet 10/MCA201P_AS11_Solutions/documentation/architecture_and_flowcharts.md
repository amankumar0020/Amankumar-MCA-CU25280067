# Architecture and Flowcharts

## Loan Approval System Architecture
```mermaid
flowchart TD
 A[Applicant inputs] --> B[Streamlit form]
 B --> C[Input validation / DataFrame]
 C --> D[Saved preprocessing + ML pipeline]
 D --> E[Loan class prediction]
 E --> F[Result displayed to user]
 G[CSV dataset] --> H[EDA and preprocessing]
 H --> I[Train/test split]
 I --> J[Compare classifiers]
 J --> K[Metrics and confusion matrix]
 J --> L[Save best pipeline with Joblib]
 L --> D
```

## Disease Prediction System Architecture
```mermaid
flowchart TD
 A[Age, temperature, symptoms] --> B[Streamlit form]
 B --> C[Feature DataFrame]
 C --> D[Saved symptom classifier]
 D --> E[Predicted dataset class]
 E --> F[Display educational result and safety notice]
 G[Synthetic symptom CSV] --> H[Cleaning and preprocessing]
 H --> I[Train/test split]
 I --> J[Compare classification algorithms]
 J --> K[Accuracy, precision, recall, F1]
 J --> L[Confusion matrix and report]
 J --> M[Save selected model with Joblib]
 M --> D
```

## Explanation
Each project has two workflows. The training workflow cleans data, separates features and target, splits the dataset, trains multiple algorithms, compares evaluation metrics and serializes the selected complete pipeline. The prediction workflow collects user input, passes it through the saved pipeline and displays the predicted class. The preprocessing steps are stored inside the loan pipeline so the same transformations are applied at prediction time.
