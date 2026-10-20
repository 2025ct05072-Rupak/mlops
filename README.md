# Heart Disease Prediction — End-to-End MLOps

## 1. Objective
Develop a reproducible machine learning classification pipeline that predicts the presence or absence of heart disease from patient health features.

## 2. Dataset
- Source: UCI Machine Learning Repository, Heart Disease dataset, ID 45.
- Acquisition: `ucimlrepo.fetch_ucirepo(id=45)`.
- Target conversion: `num = 0` becomes class 0; `num > 0` becomes class 1.
- Missing values: handled through preprocessing pipelines.
- Dataset limitations: small dataset; results may not generalize to other populations.

## 3. Exploratory Data Analysis
Include the generated histograms, correlation heatmap, class-balance chart, and missing-value summary. Explain only findings supported by the plots.

## 4. Feature Engineering and Modelling
Describe the preprocessing pipeline, numeric imputation, scaling, categorical encoding, Logistic Regression, and Random Forest.

## 5. Evaluation
Report the actual five-fold cross-validation results and holdout test metrics: accuracy, precision, recall, and ROC-AUC. Include the confusion matrix and ROC curve.

## 6. Experiment Tracking
Describe MLflow runs, parameters, metrics, models, and logged artifacts. Include a screenshot of the experiment interface.

## 7. Packaging and API
Explain model serialization using joblib and the FastAPI `/predict`, `/health`, and `/metrics` endpoints.

## 8. CI/CD
Describe the GitHub Actions workflow, linting, unit tests, model training, and uploaded workflow artifacts.

## 9. Containerization and Deployment
Document Docker build/run commands and Kubernetes deployment steps. Clearly state whether deployment was local using Minikube or hosted on a public cloud.

## 10. Monitoring
Describe request logging, latency metrics, Prometheus scraping, and any Grafana dashboard implemented.

## 11. Limitations and Conclusion
This is an educational prototype, not a clinically validated diagnostic system. Its predictions must not be used for medical decisions.

## 12. Repository
GitHub URL: [https://github.com/2025ct05072-Rupak/mlops]

## 13. Video
Video URL: []

## 14. Reproduction
Provide Python version, installation commands, dataset acquisition command, test command, training command, and API startup command.
