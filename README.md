# Diabetes Prediction Using Machine Learning

Individual ML project (Academic Year 2026-27). Binary classification of records
into diabetes-related outcome categories using the selected dataset features.

> **Disclaimer:** This is a machine learning prediction for educational purposes
> only. It is NOT a clinical diagnosis. Please consult a qualified doctor.

## Live experiment
An examiner can change values such as **Glucose** or **BMI** in the Streamlit
app and observe the predicted class.

## Project structure
| Folder / file | Purpose |
|---|---|
| `Dataset/` | Dataset CSV file |
| `Notebook/` | Jupyter notebook (EDA, training, evaluation) |
| `Source_Code/` | Python scripts (training) |
| `Model/` | Saved model (`diabetes_model.joblib`) |
| `Frontend/` | Streamlit app (`app.py`) |
| `Results/` | Plots and model comparison tables |
| `Documentation/` | Report and viva preparation |
| `n8n_workflow/` | Optional exported n8n workflow (JSON) |
| `requirements.txt` | Libraries needed to run the app |

## Run locally
```
pip install -r requirements.txt
streamlit run Frontend/app.py
```

## Results
To be filled after training and evaluation (dataset name, models compared,
metrics). No results are claimed yet.

## Author
Your name, Third Year B.Tech CSE
