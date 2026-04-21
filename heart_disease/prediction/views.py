from django.shortcuts import render
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import os
from django.conf import settings

model = None


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

def get_model():
    global model
    if model is not None:
        return model

    # Try to load dataset and train
    csv_path = os.path.join(settings.BASE_DIR, 'dataset_csv.csv')
    if os.path.exists(csv_path):
        import warnings
        warnings.filterwarnings('ignore')
        heart_data = pd.read_csv(csv_path)
        X = heart_data.drop(columns='target')
        Y = heart_data['target']
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, Y_train)
        return model
    return None

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

            input_data = (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
            
            clf = get_model()
            if clf is not None:
                input_data_as_numpy_array = np.asarray(input_data)
                input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
                prediction = clf.predict(input_data_reshaped)
                probabilities = clf.predict_proba(input_data_reshaped)[0]
                has_heart_disease = prediction[0] == 1
                confidence = round(probabilities[1 if has_heart_disease else 0] * 100, 1)
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
