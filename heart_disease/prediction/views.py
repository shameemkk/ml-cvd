from django.shortcuts import render
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import numpy as np
import os
from django.conf import settings

model = None
FEATURE_COLUMNS = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal',
]
CATEGORICAL_COLUMNS = ['cp', 'restecg', 'slope', 'thal', 'ca', 'sex', 'fbs', 'exang']
NUMERIC_COLUMNS = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']


def get_patient_suggestions(has_heart_disease):
    if has_heart_disease:
        return {
            'title': 'Suggested next steps',
            'disclaimer': 'This prediction is not a diagnosis. Please review it with a qualified doctor or cardiologist.',
            'items': [
                'Book a medical consultation soon for a proper cardiac evaluation.',
                'Seek urgent care immediately if there is chest pain, fainting, severe breathlessness, or pain spreading to the arm or jaw.',
                'Monitor blood pressure, blood sugar, cholesterol, and any recurring symptoms.',
                'Avoid smoking, limit alcohol, and reduce salty or heavily processed foods.',
                'Follow only clinician-approved exercise, medication, and follow-up plans.',
            ],
        }

    return {
        'title': 'Suggested health guidance',
        'disclaimer': 'A negative prediction does not guarantee that heart disease is absent.',
        'items': [
            'Continue routine health checkups and discuss persistent symptoms with a doctor.',
            'Maintain regular physical activity, balanced meals, and healthy sleep habits.',
            'Keep blood pressure, cholesterol, blood sugar, and weight under control.',
            'Avoid smoking and limit alcohol intake to reduce future cardiac risk.',
            'Get urgent medical help if new chest pain, severe breathlessness, or fainting appears.',
        ],
    }

def _build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERIC_COLUMNS),
            ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_COLUMNS),
        ]
    )


def _candidate_pipelines():
    pre = _build_preprocessor
    return {
        'logreg': Pipeline([('pre', pre()), ('clf', LogisticRegression(max_iter=5000, C=1.0, solver='liblinear'))]),
        'rf': Pipeline([('pre', pre()), ('clf', RandomForestClassifier(
            n_estimators=600, max_depth=None, min_samples_split=2,
            min_samples_leaf=1, random_state=42, n_jobs=-1))]),
        'gb': Pipeline([('pre', pre()), ('clf', GradientBoostingClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=3, random_state=42))]),
        'svm': Pipeline([('pre', pre()), ('clf', SVC(
            C=2.0, gamma='scale', probability=True, random_state=42))]),
    }


def get_model():
    global model
    if model is not None:
        return model

    csv_path = os.path.join(settings.BASE_DIR, 'dataset_csv.csv')
    if not os.path.exists(csv_path):
        return None

    import warnings
    warnings.filterwarnings('ignore')

    heart_data = pd.read_csv(csv_path)
    heart_data = heart_data.drop_duplicates().reset_index(drop=True)

    X = heart_data[FEATURE_COLUMNS].copy()
    Y = heart_data['target']

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, stratify=Y, random_state=42
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    candidates = _candidate_pipelines()

    scores = {}
    for name, pipe in candidates.items():
        cv_score = cross_val_score(pipe, X_train, Y_train, cv=cv, scoring='accuracy', n_jobs=-1).mean()
        scores[name] = (cv_score, pipe)

    voting = VotingClassifier(
        estimators=[(n, p) for n, (_, p) in scores.items()],
        voting='soft',
        n_jobs=-1,
    )
    voting_cv = cross_val_score(voting, X_train, Y_train, cv=cv, scoring='accuracy', n_jobs=-1).mean()
    scores['voting'] = (voting_cv, voting)

    best_name, (best_cv, best_pipe) = max(scores.items(), key=lambda kv: kv[1][0])
    best_pipe.fit(X_train, Y_train)

    test_acc = accuracy_score(Y_test, best_pipe.predict(X_test))
    print(f"[heart-model] candidate CV scores: " + ", ".join(f"{n}={s:.4f}" for n, (s, _) in scores.items()))
    print(f"[heart-model] selected={best_name}  cv={best_cv:.4f}  holdout={test_acc:.4f}  rows={len(heart_data)}")

    best_pipe.fit(X, Y)
    model = best_pipe
    return model

def home(request):
    prediction_result = None
    patient_suggestions = None
    confidence = None
    error = None

    if request.method == 'POST':
        try:
            # extract features
            age = float(request.POST.get('age', 0))
            sex = float(request.POST.get('sex', 0))
            cp = float(request.POST.get('cp', 0))
            trestbps = float(request.POST.get('trestbps', 0))
            chol = float(request.POST.get('chol', 0))
            fbs = float(request.POST.get('fbs', 0))
            restecg = float(request.POST.get('restecg', 0))
            thalach = float(request.POST.get('thalach', 0))
            exang = float(request.POST.get('exang', 0))
            oldpeak = float(request.POST.get('oldpeak', 0.0))
            slope = float(request.POST.get('slope', 0))
            ca = float(request.POST.get('ca', 0))
            thal = float(request.POST.get('thal', 0))

            input_df = pd.DataFrame([{
                'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
                'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
                'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal,
            }])[FEATURE_COLUMNS]

            clf = get_model()
            if clf is not None:
                prediction = clf.predict(input_df)
                probabilities = clf.predict_proba(input_df)[0]
                # In this dataset target=1 means HEALTHY, target=0 means HAS disease.
                has_heart_disease = prediction[0] == 0
                disease_class_index = list(clf.classes_).index(0)
                healthy_class_index = list(clf.classes_).index(1)
                confidence = round(
                    probabilities[disease_class_index if has_heart_disease else healthy_class_index] * 100,
                    1,
                )
                patient_suggestions = get_patient_suggestions(has_heart_disease)

                if not has_heart_disease:
                    prediction_result = 'The Person does not have a Heart Disease'
                else:
                    prediction_result = 'The Person has Heart Disease'
            else:
                error = "Model not trained. Please make sure 'dataset_csv.csv' is in the project root folder (next to manage.py)."
                
        except Exception as e:
            error = f"Error processing input: {str(e)}"

    return render(
        request,
        'index.html',
        {
            'prediction_result': prediction_result,
            'patient_suggestions': patient_suggestions,
            'confidence': confidence,
            'error': error,
        },
    )
