# Viva Questions and Answers

1. **What is machine learning?**  
   Machine learning is a branch of AI in which a system learns patterns from data to make predictions or decisions.

2. **What is supervised learning?**  
   It trains a model using examples that contain both input features and known target labels.

3. **What is classification?**  
   Classification predicts a category, such as Approved/Not Approved or a symptom-pattern class.

4. **What is preprocessing?**  
   It is the preparation of raw data before training, including handling missing values, duplicates and categorical values.

5. **Why handle missing values?**  
   Many algorithms cannot directly process missing values; imputation helps create a usable feature matrix.

6. **What is one-hot encoding?**  
   It converts a categorical field into binary indicator columns.

7. **Why use a train-test split?**  
   It evaluates performance on data that was not used to fit the model.

8. **What is accuracy?**  
   Correct predictions divided by all predictions.

9. **What is precision?**  
   Of the records predicted positive, the fraction that are actually positive.

10. **What is recall?**  
    Of the actual positive records, the fraction correctly identified.

11. **What is the F1-score?**  
    The harmonic mean of precision and recall.

12. **What is a confusion matrix?**  
    A table showing correct and incorrect predictions by actual and predicted class.

13. **Why compare multiple algorithms?**  
    Different algorithms learn patterns differently; comparison helps select a model based on validation metrics.

14. **What is Random Forest?**  
    An ensemble that combines predictions from multiple decision trees.

15. **What is overfitting?**  
    When a model learns training examples too closely and performs worse on unseen data.

16. **Why use Joblib?**  
    It saves and loads trained Python models and pipelines.

17. **What is Streamlit?**  
    A Python framework for building interactive data applications quickly.

18. **What is a pipeline?**  
    A sequence of preprocessing and model steps executed together.

19. **What are the loan project inputs?**  
    Applicant details such as income, education, employment, loan amount, credit history and marital status.

20. **What are the disease project inputs?**  
    Age, temperature and selected symptoms in the supplied educational dataset.

21. **Can the disease project diagnose a patient?**  
    No. It uses synthetic teaching data and is not clinically validated or suitable for diagnosis or treatment.

22. **Can the loan model make an actual bank decision?**  
    No. It is a classroom demonstration trained on synthetic data, not a lending policy or credit assessment service.

23. **What is data leakage?**  
    It occurs when information unavailable at prediction time improperly enters model training or evaluation.

24. **How can the projects be improved?**  
    Use verified, representative data; add cross-validation, fairness checks, monitoring, input validation and secure deployment.

25. **What is the purpose of the project?**  
    To demonstrate an end-to-end ML workflow: preprocessing, algorithm comparison, evaluation, model saving and a prediction interface.
