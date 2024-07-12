import logging, base64
import os, pandas as pd,io
import joblib,json
from django.contrib.auth import logout
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.hashers import make_password, check_password
from django.shortcuts import render, redirect
from rest_framework import status, views, viewsets
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.views import APIView
from .serializers import PredictionSerializer_LungCancer, PredictionSerializer_BreastCancer, PredictionSerializer_Depression, PredictionSerializer_Diabetes, PredictionSerializer_HeartDisease, PredictionSerializer_SkinDisease
from django.http import JsonResponse, HttpResponse
from django.views import View
from .serializers import PredictionSerializer_Stroke, PredictionSerializer_KidneyDisease, PredictionSerializer_ParkinssonDisease
from matplotlib.dates import DateFormatter
logger = logging.getLogger('django')
from .models import DiseaseTest_Coll,SymptomMedicine_Coll, SymptomMedicine, MedicalTest
from .forms import SymptomForm, DiseaseForm
from rest_framework.pagination import PageNumberPagination
from rest_framework.generics import ListAPIView
import pandas as pd
import openpyxl
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from .models import DiseasePrediction,Prediction
#from .serializers import DiseasePredictionSerializer
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from .models import User
from .serializers import UserSignupSerializer, UserLoginSerializer
from rest_framework_simplejwt.tokens import RefreshToken
from django.contrib.auth.hashers import check_password
from rest_framework.permissions import AllowAny
from django.contrib.auth import get_user_model
from rest_framework.exceptions import AuthenticationFailed
from django.contrib.auth import authenticate, login
import jwt,time
from mongoengine import DoesNotExist
from mongoengine.queryset.visitor import Q





# LUNGCANCER
class PredictAndPlotLungCancerView(APIView):
    last_prediction = None
    predictions = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Load the model
        model_path = os.path.join(os.path.dirname(__file__), 'model', 'model_lung_cancer.pkl')
        encoder_path = os.path.join(os.path.dirname(__file__), 'model', 'model_lung_cancer_encoder.pkl')

        try:
            self.model = joblib.load(model_path)
            self.encoder = joblib.load(encoder_path)
            logger.info("Model and encoder loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model or encoder: {e}")
            self.model = None
            self.encoder = None

    def post(self, request):
        logger.debug(f"Request data: {request.data}")
        serializer = PredictionSerializer_LungCancer(data=request.data)
        if serializer.is_valid():
            if self.model is None or self.encoder is None:
                logger.error("Model or encoder not loaded.")
                return Response({'error': 'Model or encoder not loaded'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            data = serializer.validated_data
            symptoms = {
                'GENDER': data['GENDER'],
                'AGE': data['AGE'],
                'SMOKING': 1 if data['SMOKING'].upper() == 'YES' else 0,
                'YELLOW_FINGERS': 1 if data['YELLOW_FINGERS'].upper() == 'YES' else 2,
                'ANXIETY': 1 if data['ANXIETY'].upper() == 'YES' else 0,
                'PEER_PRESSURE': 1 if data['PEER_PRESSURE'].upper() == 'YES' else 2,
                'CHRONIC_DISEASE': 1 if data['CHRONIC_DISEASE'].upper() == 'YES' else 2,
                'FATIGUE': 1 if data['FATIGUE'].upper() == 'YES' else 2,
                'ALLERGY': 1 if data['ALLERGY'].upper() == 'YES' else 2,
                'WHEEZING': 1 if data['WHEEZING'].upper() == 'YES' else 2,
                'ALCOHOL_CONSUMING': 1 if data['ALCOHOL_CONSUMING'].upper() == 'YES' else 2,
                'COUGHING': 1 if data['COUGHING'].upper() == 'YES' else 2,
                'SHORTNESS_OF_BREATH': 1 if data['SHORTNESS_OF_BREATH'].upper() == 'YES' else 2,
                'SWALLOWING_DIFFICULTY': 1 if data['SWALLOWING_DIFFICULTY'].upper() == 'YES' else 2,
                'CHEST_PAIN': 1 if data['CHEST_PAIN'].upper() == 'YES' else 2
            }

            symptoms_df = pd.DataFrame([symptoms])

            try:
                symptoms_encoded = self.encoder.transform(symptoms_df)
                symptoms = symptoms_encoded.toarray() if hasattr(symptoms_encoded, 'toarray') else symptoms_encoded
                prediction_lungcancer = self.model.predict(symptoms)[0]
                logger.debug(f"Prediction result: {prediction_lungcancer}")
                self.__class__.last_prediction = prediction_lungcancer
                self.__class__.predictions.append((datetime.now(), prediction_lungcancer))

                prediction_label = 'YES' if prediction_lungcancer >= 0.5 else 'NO'

                email = data['EMAIL']

                self.save_prediction_results(email, prediction_label)

                chart_image = self.generate_chart()

                context = {
                    'prediction': prediction_lungcancer,
                    'prediction_label': prediction_label,
                    'chart_image': chart_image
                }
                return render(request, 'Int_app/lung_cancer_results.html', context)
            except Exception as e:
                logger.error(f"Prediction failed: {e}")
                return Response({'error': f'Prediction failed: {e}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        else:
            logger.error(f"Serializer errors: {serializer.errors}")
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def generate_chart(self):
        # Create a DataFrame from the predictions
        df = pd.DataFrame(self.__class__.predictions, columns=['date', 'prediction'])
        logger.debug(f"Prediction DataFrame: {df}")

        if df.empty:
            return None

        # Plot the data
        fig, ax = plt.subplots()
        ax.plot(df['date'], df['prediction'], marker='o', linestyle='-')
        ax.set_xlabel('Date')
        ax.set_ylabel('Prediction')
        ax.set_title('Lung Cancer Predictions Over Time')

        # Format the x-axis to show dates in a vertical manner
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.xticks(rotation=90)

        # Save the plot to a BytesIO object and encode it as base64
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        chart_image = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)
        return chart_image

    @staticmethod
    def save_prediction_results(email, last_prediction):
        try:
            # Create a prediction entry
            prediction_entry = Prediction(
                disease='Lung Cancer',
                prediction_value=last_prediction,
                timestamp=datetime.now()
            )

            # Check if a DiseasePrediction document already exists for the email
            disease_prediction = DiseasePrediction.objects(email=email).first()

            if disease_prediction:
                # Check if there is an existing prediction for depression on the same date
                today = datetime.now().date()
                existing_prediction = None
                for pred in disease_prediction.predictions:
                    if pred.disease == 'Lung Cancer' and pred.timestamp.date() == today:
                        existing_prediction = pred
                        break

                if existing_prediction:
                    # Remove the existing prediction
                    disease_prediction.predictions.remove(existing_prediction)

                # Append the new prediction
                disease_prediction.predictions.append(prediction_entry)
                disease_prediction.save()
            else:
                # Create a new DiseasePrediction document
                new_disease_prediction = DiseasePrediction(
                    email=email,
                    predictions=[prediction_entry]
                )
                new_disease_prediction.save()

            logger.info("Prediction saved successfully.")
        except Exception as e:
            logger.error(f"Error saving prediction: {e}")

def prediction_LungCancer_form(request):
    print("reaching..")
    return render(request, 'Int_app/demo_page.html')



# BREAST CANCER

class PredictAndPlotBreastCancerView(APIView):
    last_prediction = None
    predictions = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Load the model and encoder
        model_path = os.path.join(os.path.dirname(__file__), 'model', 'model_breast_cancer.pkl')
        encoder_path = os.path.join(os.path.dirname(__file__), 'model', 'model_breast_cancer_encoder.pkl')

        try:
            self.model = joblib.load(model_path)
            self.encoder = joblib.load(encoder_path)
            logger.info("Model and encoder loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model or encoder: {e}")
            self.model = None
            self.encoder = None

    def post(self, request):
        logger.debug(f"Request data: {request.data}")
        serializer = PredictionSerializer_BreastCancer(data=request.data)
        if serializer.is_valid():
            if self.model is None or self.encoder is None:
                logger.error("Model or encoder not loaded.")
                return Response({'error': 'Model or encoder not loaded'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            data = serializer.validated_data
            symptoms = {
                'Age': data['AGE'],
                'TStage': data['T_STAGE'],
                'NStage': data['N_STAGE'],
                '6Stage': data['FINAL_STAGE'],
                'Grade': data['GRADE'],
                'Examined_Node': data['REGIONAL_NODE_EXAMINED'],
                'Positive_Node': data['REGIONAL_NODE_POSITIVE'],
                'SurvivalMonths': data['SURVIVAL_MONTHS'],
                'Race': data['RACE'],
                'MaritalStat': data['MARITAL_STAT'],
                'Differentiated': data['DIFFERENTIATED'],
                'AStage': data['A_STAGE'],
                'LiveStatus': data['ALIVE_STAT'],
                'Estrogen': data['ESTROGEN'],
                'Progesterone': data['PROGESTERONE']
            }

            symptoms_df = pd.DataFrame([symptoms])
            logger.debug(f"Symptoms DataFrame before encoding: {symptoms_df}")

            try:
                # Apply one-hot encoding
                symptoms_encoded = self.encoder.transform(symptoms_df)
                logger.debug(f"Symptoms after encoding: {symptoms_encoded}")

                # Convert to numpy array if necessary
                symptoms = symptoms_encoded.toarray() if hasattr(symptoms_encoded, 'toarray') else symptoms_encoded

                # Make prediction
                prediction_breastcancer = self.model.predict(symptoms)[0]
                logger.debug(f"Prediction result: {prediction_breastcancer}")
                self.__class__.last_prediction = prediction_breastcancer
                self.__class__.predictions.append((datetime.now(), prediction_breastcancer))

                email = data['EMAIL']

                self.save_prediction_results(email, prediction_breastcancer)


                chart_image = self.generate_chart()

                context = {
                    'prediction': prediction_breastcancer,

                    'chart_image': chart_image
                }
                return render(request, 'Int_app/breast_cancer_results.html', context)
            except Exception as e:
                logger.error(f"Prediction failed: {e}")
                return Response({'error': f'Prediction failed: {e}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        else:
            logger.error(f"Serializer errors: {serializer.errors}")
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def generate_chart(self):
        # Create a DataFrame from the predictions
        df = pd.DataFrame(self.__class__.predictions, columns=['date', 'prediction'])
        logger.debug(f"Prediction DataFrame: {df}")

        if df.empty:
            return None

        # Plot the data
        fig, ax = plt.subplots()
        ax.plot(df['date'], df['prediction'], marker='o', linestyle='-')
        ax.set_xlabel('Date')
        ax.set_ylabel('Prediction')
        ax.set_title('Breast Cancer Predictions Over Time')

        # Format the x-axis to show dates in a vertical manner
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.xticks(rotation=90)

        # Save the plot to a BytesIO object and encode it as base64
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        chart_image = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)
        return chart_image

    @staticmethod
    def save_prediction_results(email, last_prediction):
        try:
            # Create a prediction entry
            prediction_entry = Prediction(
                disease='Breast Cancer',
                prediction_value=last_prediction,
                timestamp=datetime.now()
            )

            # Check if a DiseasePrediction document already exists for the email
            disease_prediction = DiseasePrediction.objects(email=email).first()

            if disease_prediction:
                # Check if there is an existing prediction for depression on the same date
                today = datetime.now().date()
                existing_prediction = None
                for pred in disease_prediction.predictions:
                    if pred.disease == 'Breast Cancer' and pred.timestamp.date() == today:
                        existing_prediction = pred
                        break

                if existing_prediction:
                    # Remove the existing prediction
                    disease_prediction.predictions.remove(existing_prediction)

                # Append the new prediction
                disease_prediction.predictions.append(prediction_entry)
                disease_prediction.save()
            else:
                # Create a new DiseasePrediction document
                new_disease_prediction = DiseasePrediction(
                    email=email,
                    predictions=[prediction_entry]
                )
                new_disease_prediction.save()

            logger.info("Prediction saved successfully.")
        except Exception as e:
            logger.error(f"Error saving prediction: {e}")
def prediction_BreastCancer_form(request):
    print("reaching..")
    return render(request, 'Int_app/Breast_Cancer.html')



# DEPRESSION

class PredictAndPlotDepressionView(APIView):
    last_prediction = None
    predictions = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Load the model and scaler
        model_path = os.path.join(os.path.dirname(__file__), 'model', 'depression.pkl')
        scalar_path = os.path.join(os.path.dirname(__file__), 'model', 'depression_scalar.pkl')

        try:
            self.model = joblib.load(model_path)
            self.scalar = joblib.load(scalar_path)
            logger.info("Model and scaler loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model or scaler: {e}")
            self.model = None
            self.scalar = None

    def post(self, request):
        logger.debug(f"Request data: {request.data}")
        serializer = PredictionSerializer_Depression(data=request.data)
        if serializer.is_valid():
            if self.model is None or self.scalar is None:
                logger.error("Model or scaler not loaded.")
                return Response({'error': 'Model or scaler not loaded'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            data = serializer.validated_data
            symptoms = {
                'SLEEP': data['SLEEP'],
                'APPETITE': data['APPETITE'],
                'INTEREST': data['INTEREST'],
                'FATIGUE': data['FATIGUE'],
                'WORTHLESSNESS': data['WORTHLESSNESS'],
                'CONCENTRATION': data['CONCENTRATION'],
                'AGITATION': data['AGITATION'],
                'SUICIDAL_THOUGHTS': data['SUICIDAL_THOUGHTS'],
                'SLEEP_DISORDERS': data['SLEEP_DISORDERS'],
                'AGGRESSION': data['AGGRESSION'],
                'PANIC_ATTACKS': data['PANIC_ATTACKS'],
                'HOPELESSNESS': data['HOPELESSNESS'],
                'RESTLESSNESS': data['RESTLESSNESS'],
                'LOW_ENERGY': data['LOW_ENERGY']
            }

            symptoms_df = pd.DataFrame([symptoms])
            logger.debug(f"Symptoms DataFrame before scaling: {symptoms_df}")

            try:
                symptoms_scaled = self.scalar.transform(symptoms_df)
                symptoms = symptoms_scaled.toarray() if hasattr(symptoms_scaled, 'toarray') else symptoms_scaled
                prediction_depression = self.model.predict(symptoms)[0]
                logger.debug(f"Prediction result: {prediction_depression}")
                self.__class__.last_prediction = prediction_depression
                self.__class__.predictions.append((datetime.now(), prediction_depression))

                prediction_label = 'YES' if prediction_depression >= 1.5 else 'NO'

                email = data['EMAIL']


                self.save_prediction_results(email, prediction_label)

                chart_image = self.generate_chart()

                context = {
                    'prediction': prediction_depression,
                    'prediction_label': prediction_label,
                    'chart_image': chart_image
                }
                return render(request, 'Int_app/depression_results.html', context)
            except Exception as e:
                logger.error(f"Prediction failed: {e}")
                return Response({'error': f'Prediction failed: {e}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        else:
            logger.error(f"Serializer errors: {serializer.errors}")
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def generate_chart(self):
        # Create a DataFrame from the predictions
        df = pd.DataFrame(self.__class__.predictions, columns=['date', 'prediction'])
        logger.debug(f"Prediction DataFrame: {df}")

        if df.empty:
            return None

        # Plot the data
        fig, ax = plt.subplots()
        ax.plot(df['date'], df['prediction'], marker='o', linestyle='-')
        ax.set_xlabel('Date')
        ax.set_ylabel('Prediction')
        ax.set_title('Depression Predictions Over Time')

        # Format the x-axis to show dates in a vertical manner
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.xticks(rotation=90)

        # Save the plot to a BytesIO object and encode it as base64
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        chart_image = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)
        return chart_image

    @staticmethod
    def save_prediction_results(email, last_prediction):
        try:
            # Create a prediction entry
            prediction_entry = Prediction(
                disease='Depression',
                prediction_value=last_prediction,
                timestamp=datetime.now()
            )

            # Check if a DiseasePrediction document already exists for the email
            disease_prediction = DiseasePrediction.objects(email=email).first()

            if disease_prediction:
                # Check if there is an existing prediction for depression on the same date
                today = datetime.now().date()
                existing_prediction = None
                for pred in disease_prediction.predictions:
                    if pred.disease == 'Depression' and pred.timestamp.date() == today:
                        existing_prediction = pred
                        break

                if existing_prediction:
                    # Remove the existing prediction
                    disease_prediction.predictions.remove(existing_prediction)

                # Append the new prediction
                disease_prediction.predictions.append(prediction_entry)
                disease_prediction.save()
            else:
                # Create a new DiseasePrediction document
                new_disease_prediction = DiseasePrediction(
                    email=email,
                    predictions=[prediction_entry]
                )
                new_disease_prediction.save()

            logger.info("Prediction saved successfully.")
        except Exception as e:
            logger.error(f"Error saving prediction: {e}")

def prediction_Depression_form(request):
    print("reaching..")
    return render(request, 'Int_app/Depression.html')



# DIABETES

class PredictAndPlotDiabetesView(APIView):
    last_prediction = None
    predictions = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Load the model and scaler
        model_path = os.path.join(os.path.dirname(__file__), 'model', 'model_diabetes.pkl')
        scaler_path = os.path.join(os.path.dirname(__file__), 'model', 'model_diabetes_scalar.pkl')

        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            logger.info("Model and scaler loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model or scaler: {e}")
            self.model = None
            self.scaler = None

    def post(self, request):
        logger.debug(f"Request data: {request.data}")
        serializer = PredictionSerializer_Diabetes(data=request.data)
        if serializer.is_valid():
            if self.model is None or self.scaler is None:
                logger.error("Model or scaler not loaded.")
                return Response({'error': 'Model or scaler not loaded'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            data = serializer.validated_data
            symptoms = {
                'PREGNANCY': [data['PREGNANCY']],
                'GLUCOSE': [data['GLUCOSE']],
                'BLOOD_PRESSURE': [data['BLOOD_PRESSURE']],
                'SKIN_THICKNESS': [data['SKIN_THICKNESS']],
                'INSULIN': [data['INSULIN']],
                'BMI': [data['BMI']],
                'DIABETES_FUNCTION': [data['DIABETES_FUNCTION']],
                'AGE': [data['AGE']],
            }

            symptoms_df = pd.DataFrame(symptoms)
            logger.debug(f"Symptoms DataFrame before scaling: {symptoms_df}")

            try:
                symptoms_scaled = self.scaler.transform(symptoms_df)
                symptoms = symptoms_scaled.toarray() if hasattr(symptoms_scaled, 'toarray') else symptoms_scaled
                prediction_diabetes = self.model.predict(symptoms)[0]
                logger.debug(f"Prediction result: {prediction_diabetes}")
                self.__class__.last_prediction = prediction_diabetes
                self.__class__.predictions.append((datetime.now(), prediction_diabetes))

                prediction_label = 'YES' if prediction_diabetes >= 0.6 else 'NO'

                email = data['EMAIL']

                self.save_prediction_results(email, prediction_label)

                chart_image = self.generate_chart()

                context = {
                    'prediction': prediction_diabetes,
                    'prediction_label': prediction_label,
                    'chart_image': chart_image
                }
                return render(request, 'Int_app/diabetes_results.html', context)
            except Exception as e:
                logger.error(f"Prediction failed: {e}")
                return Response({'error': f'Prediction failed: {e}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        else:
            logger.error(f"Serializer errors: {serializer.errors}")
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def generate_chart(self):
        # Create a DataFrame from the predictions
        df = pd.DataFrame(self.__class__.predictions, columns=['date', 'prediction'])
        logger.debug(f"Prediction DataFrame: {df}")

        if df.empty:
            return None

        # Plot the data
        fig, ax = plt.subplots()
        ax.plot(df['date'], df['prediction'], marker='o', linestyle='-')
        ax.set_xlabel('Date')
        ax.set_ylabel('Prediction')
        ax.set_title('Diabetes Predictions Over Time')

        # Format the x-axis to show dates in a vertical manner
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.xticks(rotation=90)

        # Save the plot to a BytesIO object and encode it as base64
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        chart_image = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)
        return chart_image

    @staticmethod
    def save_prediction_results(email, last_prediction):
        try:
            # Create a prediction entry
            prediction_entry = Prediction(
                disease='Diabetes',
                prediction_value=last_prediction,
                timestamp=datetime.now()
            )

            # Check if a DiseasePrediction document already exists for the email
            disease_prediction = DiseasePrediction.objects(email=email).first()

            if disease_prediction:
                # Check if there is an existing prediction for depression on the same date
                today = datetime.now().date()
                existing_prediction = None
                for pred in disease_prediction.predictions:
                    if pred.disease == 'Diabetes' and pred.timestamp.date() == today:
                        existing_prediction = pred
                        break

                if existing_prediction:
                    # Remove the existing prediction
                    disease_prediction.predictions.remove(existing_prediction)

                # Append the new prediction
                disease_prediction.predictions.append(prediction_entry)
                disease_prediction.save()
            else:
                # Create a new DiseasePrediction document
                new_disease_prediction = DiseasePrediction(
                    email=email,
                    predictions=[prediction_entry]
                )
                new_disease_prediction.save()

            logger.info("Prediction saved successfully.")
        except Exception as e:
            logger.error(f"Error saving prediction: {e}")

def prediction_Diabetes_form(request):
    print("reaching..")
    return render(request, 'Int_app/Diabetes.html')


# HEART_DISEASE

class PredictAndPlotHeartDiseaseView(APIView):
    last_prediction = None
    predictions = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Load the model and encoder
        model_path = os.path.join(os.path.dirname(__file__), 'model', 'model_heart_disease.pkl')
        encoder_path = os.path.join(os.path.dirname(__file__), 'model', 'model_heart_disease_encoder.pkl')

        try:
            self.model = joblib.load(model_path)
            self.encoder = joblib.load(encoder_path)
            logger.info("Model and encoder loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model or encoder: {e}")
            self.model = None
            self.encoder = None

    def post(self, request):
        logger.debug(f"Request data: {request.data}")
        serializer = PredictionSerializer_HeartDisease(data=request.data)
        if serializer.is_valid():
            if self.model is None or self.encoder is None:
                logger.error("Model or encoder not loaded.")
                return Response({'error': 'Model or encoder not loaded'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            data = serializer.validated_data
            symptoms = {
                'Age': data['AGE'],
                'Sex': data['SEX'],
                'Chest_Pain': data['CHEST_PAIN'],
                'Resting_BP': data['RESTING_BP'],
                'Cholestoral': data['CHOLESTORAL'],
                'Fasting_BP': data['FASTING_BP'],
                'Rest_ecg': data['REST_ECG'],
                'Max_heartrate': data['MAX_HEARTRATE'],
                'Excercise': data['EXCERCISE'],
                'Old_Peak': data['OLD_PEAK'],
                'Slope': data['SLOPING'],
                'Vessels': data['VESSELS'],
                'Thalassemia': data['THALASSEMIA']
            }

            symptoms_df = pd.DataFrame([symptoms])
            logger.debug(f"Symptoms DataFrame before encoding: {symptoms_df}")

            try:
                # Apply one-hot encoding
                symptoms_encoded = self.encoder.transform(symptoms_df)
                logger.debug(f"Symptoms after encoding: {symptoms_encoded}")

                # Convert to numpy array if necessary
                symptoms = symptoms_encoded.toarray() if hasattr(symptoms_encoded, 'toarray') else symptoms_encoded

                # Make prediction
                prediction_heart_disease = self.model.predict(symptoms)[0]
                logger.debug(f"Prediction result: {prediction_heart_disease}")
                self.__class__.last_prediction = prediction_heart_disease
                self.__class__.predictions.append((datetime.now(), prediction_heart_disease))
                prediction_label = 'YES' if prediction_heart_disease >= 0.0 else 'NO'

                email = data['EMAIL']

                self.save_prediction_results(email, prediction_label)

                chart_image = self.generate_chart()

                context = {
                    'prediction': prediction_heart_disease,
                    'prediction_label': prediction_label,
                    'chart_image': chart_image
                }
                return render(request, 'Int_app/heart_results.html', context)
            except Exception as e:
                logger.error(f"Prediction failed: {e}")
                return Response({'error': f'Prediction failed: {e}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        else:
            logger.error(f"Serializer errors: {serializer.errors}")
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def generate_chart(self):
        # Create a DataFrame from the predictions
        df = pd.DataFrame(self.__class__.predictions, columns=['date', 'prediction'])
        logger.debug(f"Prediction DataFrame: {df}")

        if df.empty:
            return None

        # Plot the data
        fig, ax = plt.subplots()
        ax.plot(df['date'], df['prediction'], marker='o', linestyle='-')
        ax.set_xlabel('Date')
        ax.set_ylabel('Prediction')
        ax.set_title('Heart Disease Predictions Over Time')

        # Format the x-axis to show dates in a vertical manner
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.xticks(rotation=90)

        # Save the plot to a BytesIO object and encode it as base64
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        chart_image = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)
        return chart_image

    @staticmethod
    def save_prediction_results(email, last_prediction):
        try:
            # Create a prediction entry
            prediction_entry = Prediction(
                disease='Heart Disease',
                prediction_value=last_prediction,
                timestamp=datetime.now()
            )

            # Check if a DiseasePrediction document already exists for the email
            disease_prediction = DiseasePrediction.objects(email=email).first()

            if disease_prediction:
                # Check if there is an existing prediction for depression on the same date
                today = datetime.now().date()
                existing_prediction = None
                for pred in disease_prediction.predictions:
                    if pred.disease == 'Heart Disease' and pred.timestamp.date() == today:
                        existing_prediction = pred
                        break

                if existing_prediction:
                    # Remove the existing prediction
                    disease_prediction.predictions.remove(existing_prediction)

                # Append the new prediction
                disease_prediction.predictions.append(prediction_entry)
                disease_prediction.save()
            else:
                # Create a new DiseasePrediction document
                new_disease_prediction = DiseasePrediction(
                    email=email,
                    predictions=[prediction_entry]
                )
                new_disease_prediction.save()

            logger.info("Prediction saved successfully.")
        except Exception as e:
            logger.error(f"Error saving prediction: {e}")

def prediction_HeartDisease_form(request):
    print("reaching..")
    return render(request, 'Int_app/Heart_Disease.html')



# SKIN DISEASE

class PredictAndPlotSkinDiseaseView(APIView):
    last_prediction = None
    predictions = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Load the model and imputer
        model_path = os.path.join(os.path.dirname(__file__), 'model', 'model_skin_disease.pkl')
        imputer_path = os.path.join(os.path.dirname(__file__), 'model', 'model_skin_disease_imputer.pkl')

        try:
            self.model = joblib.load(model_path)
            self.imputer = joblib.load(imputer_path)
            if not hasattr(self.imputer, 'transform'):
                raise ValueError("Loaded imputer does not have a 'transform' method")
            logger.info("Model and imputer loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model or imputer: {e}")
            self.model = None
            self.imputer = None

    def post(self, request):
        logger.debug(f"Request data: {request.data}")
        serializer = PredictionSerializer_SkinDisease(data=request.data)
        if serializer.is_valid():
            if self.model is None or self.imputer is None:
                logger.error("Model or imputer not loaded.")
                return Response({'error': 'Model or imputer not loaded'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            data = serializer.validated_data
            symptoms = {
                'ERYTHEMA': data['ERYTHEMA'],
                'SCALING': data['SCALING'],
                'DEFINITE_BORDERS': data['DEFINITE_BORDERS'],
                'ITCHING': data['ITCHING'],
                'KOEBNER_PHENOMENON': data['KOEBNER_PHENOMENON'],
                'POLYGONAL_PAPULES': data['POLYGONAL_PAPULES'],
                'FOLLICULAR_PAPULES': data['FOLLICULAR_PAPULES'],
                'ORAL_MUCOSAL_INVOLVEMENT': data['ORAL_MUCOSAL_INVOLVEMENT'],
                'KNEE_AND_ELBOW_INVOLVEMENT': data['KNEE_AND_ELBOW_INVOLVEMENT'],
                'SCALP_INVOLVEMENT': data['SCALP_INVOLVEMENT'],
                'FAMILY_HISTORY': data['FAMILY_HISTORY'],
                'MELANIN_INCONTINENCE': data['MELANIN_INCONTINENCE'],
                'EOSINOPHILS_INFILTRATE': data['EOSINOPHILS_INFILTRATE'],
                'PNL_INFILTRATE': data['PNL_INFILTRATE'],
                'FIBROSIS_PAPILLARY_DERMIS': data['FIBROSIS_PAPILLARY_DERMIS'],
                'EXOCYTOSIS': data['EXOCYTOSIS'],
                'ACANTHOSIS': data['ACANTHOSIS'],
                'HYPERKERATOSIS': data['HYPERKERATOSIS'],
                'PARAKERATOSIS': data['PARAKERATOSIS'],
                'CLUBBING_RETE_RIDGES': data['CLUBBING_RETE_RIDGES'],
                'ELONGATION_RETE_RIDGES': data['ELONGATION_RETE_RIDGES'],
                'THINNING_SUPRAPAPILLARY_EPDERMIS': data['THINNING_SUPRAPAPILLARY_EPDERMIS'],
                'SPONGIFORM_PUSTULE': data['SPONGIFORM_PUSTULE'],
                'MUNRO_MICROABCESS': data['MUNRO_MICROABCESS'],
                'FOCAL_HYPERGRANULOSIS': data['FOCAL_HYPERGRANULOSIS'],
                'DISAPPEARANCE_GRANULAR_LAYER': data['DISAPPEARANCE_GRANULAR_LAYER'],
                'VACUOLISATION_DAMAGE_BASAL_LAYER': data['VACUOLISATION_DAMAGE_BASAL_LAYER'],
                'SPONGIOSIS': data['SPONGIOSIS'],
                'SAW_TOOTH_APPEARANCE_RETSES': data['SAW_TOOTH_APPEARANCE_RETSES'],
                'FOLLICULAR_HORN_PLUG': data['FOLLICULAR_HORN_PLUG'],
                'PERIFOLLICULAR_PARAKERATOSIS': data['PERIFOLLICULAR_PARAKERATOSIS'],
                'INFLAMMATORY_MONONUCLEAR_INFILTRATE': data['INFLAMMATORY_MONONUCLEAR_INFILTRATE'],
                'BAND_LIKE_INFILTRATE': data['BAND_LIKE_INFILTRATE'],
                'AGE': data['AGE'],
            }

            symptoms_df = pd.DataFrame([symptoms])

            try:
                # Extract AGE column for imputation
                age_column = symptoms_df['AGE'].values.reshape(-1, 1)

                # Apply the imputer to AGE column
                age_imputed = self.imputer.transform(age_column)

                # Combine imputed AGE with the rest of the DataFrame
                symptoms_imputed = symptoms_df.drop(columns=['AGE'])
                symptoms_imputed['AGE'] = age_imputed

                # Make a prediction
                prediction_skin_disease = self.model.predict(symptoms_imputed)[0]
                logger.debug(f"Prediction result: {prediction_skin_disease}")
                self.__class__.last_prediction = prediction_skin_disease
                self.__class__.predictions.append((datetime.now(), prediction_skin_disease))

                if prediction_skin_disease <= 2.2:
                    prediction_label = 'Psoriasis'
                elif 2.2 < prediction_skin_disease <= 3.2:
                    prediction_label = 'Seborrheic Dermatitis'
                elif 3.2 < prediction_skin_disease <= 4.2:
                    prediction_label = 'Lichen Planus'
                elif 4.2 < prediction_skin_disease <= 5.2:
                    prediction_label = 'Pityriasis Rosea'
                elif 5.2 < prediction_skin_disease <= 6.2:
                    prediction_label = 'Chronic Dermatitis'
                elif 6.2 < prediction_skin_disease <= 7.2:
                    prediction_label = 'Pityriasis Rubra Pilaris'

                email = data['EMAIL']

                self.save_prediction_results(email, prediction_label)

                chart_image = self.generate_chart()

                context = {
                    'prediction': prediction_skin_disease,
                    'prediction_label': prediction_label,
                    'chart_image': chart_image
                }
                return render(request, 'Int_app/skin_results.html', context)
            except Exception as e:
                logger.error(f"Prediction failed: {e}")
                return Response({'error': f'Prediction failed: {e}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        else:
            logger.error(f"Serializer errors: {serializer.errors}")
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def generate_chart(self):
        # Create a DataFrame from the predictions
        df = pd.DataFrame(self.__class__.predictions, columns=['date', 'prediction'])
        logger.debug(f"Prediction DataFrame: {df}")

        if df.empty:
            return None

        # Plot the data
        fig, ax = plt.subplots()
        ax.plot(df['date'], df['prediction'], marker='o', linestyle='-')
        ax.set_xlabel('Date')
        ax.set_ylabel('Prediction')
        ax.set_title('Skin Disease Predictions Over Time')

        # Format the x-axis to show dates in a vertical manner
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.xticks(rotation=90)

        # Save the plot to a BytesIO object and encode it as base64
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        chart_image = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)
        return chart_image

    @staticmethod
    def save_prediction_results(email, last_prediction):
        try:
            # Create a prediction entry
            prediction_entry = Prediction(
                disease='Skin Disease',
                prediction_value=last_prediction,
                timestamp=datetime.now()
            )

            # Check if a DiseasePrediction document already exists for the email
            disease_prediction = DiseasePrediction.objects(email=email).first()

            if disease_prediction:
                # Check if there is an existing prediction for depression on the same date
                today = datetime.now().date()
                existing_prediction = None
                for pred in disease_prediction.predictions:
                    if pred.disease == 'Skin Disease' and pred.timestamp.date() == today:
                        existing_prediction = pred
                        break

                if existing_prediction:
                    # Remove the existing prediction
                    disease_prediction.predictions.remove(existing_prediction)

                # Append the new prediction
                disease_prediction.predictions.append(prediction_entry)
                disease_prediction.save()
            else:
                # Create a new DiseasePrediction document
                new_disease_prediction = DiseasePrediction(
                    email=email,
                    predictions=[prediction_entry]
                )
                new_disease_prediction.save()

            logger.info("Prediction saved successfully.")
        except Exception as e:
            logger.error(f"Error saving prediction: {e}")


def prediction_SkinDisease_form(request):
    print("reaching..")
    return render(request, 'Int_app/Skin_Disease.html')




#STROKE

class PredictAndPlotStrokeView(APIView):
    last_prediction = None
    predictions = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Load the model, encoder, imputer
        model_path = os.path.join(os.path.dirname(__file__), 'model', 'model_stroke.pkl')
        encoder_path = os.path.join(os.path.dirname(__file__), 'model', 'model_stroke_encoder.pkl')
        imputer_path = os.path.join(os.path.dirname(__file__), 'model', 'model_stroke_imputer.pkl')

        try:
            self.model = joblib.load(model_path)
            self.encoder = joblib.load(encoder_path)
            self.imputer = joblib.load(imputer_path)
            logger.info("Model, encoder, and imputer loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model, encoder, or imputer: {e}")
            self.model = None
            self.encoder = None
            self.imputer = None

    def post(self, request):
        logger.debug(f"Request data: {request.data}")
        serializer = PredictionSerializer_Stroke(data=request.data)

        if serializer.is_valid():
            if self.model is None or self.imputer is None or self.encoder is None:
                logger.error("Model or imputer not loaded.")
                return Response({'error': 'Model or imputer not loaded'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            data = serializer.validated_data

            symptoms = {
                'gender': data['GENDER'],
                'age': data['AGE'],
                'hypertension': data['HYPERTENSION'],
                'heartdisease': data['HEART_DISEASE'],
                'maritalstat': data['MARITAL_STATUS'],
                'worktype': data['EMPLOYED'],
                'residencetype': data['RESIDENCE'],
                'glucoselevel': data['AVG_GLUCOSE'],
                'bmi': data['BMI'],
                'smokingstat': data['SMOKING_STAT']
            }
            # Convert to DataFrame
            symptoms_df = pd.DataFrame([symptoms])
            logger.debug(f"Symptoms DataFrame before imputation: {symptoms_df}")

            try:
                # Handle missing values
                symptoms_df[['bmi']] = self.imputer.transform(symptoms_df[['bmi']])
                logger.debug(f"Symptoms DataFrame after imputation: {symptoms_df}")

                # Apply one-hot encoding
                symptoms_encoded = self.encoder.transform(symptoms_df)
                logger.debug(f"Symptoms after encoding: {symptoms_encoded}")

                # Convert to numpy array if necessary
                symptoms = symptoms_encoded.toarray() if hasattr(symptoms_encoded, 'toarray') else symptoms_encoded

                # Make prediction
                prediction_Stroke = self.model.predict(symptoms)[0]
                logger.debug(f"Prediction result: {prediction_Stroke}")
                self.__class__.last_prediction = prediction_Stroke
                self.__class__.predictions.append((datetime.now(), prediction_Stroke))
                prediction_label = 'YES' if prediction_Stroke >= 0.5 else 'NO'

                email = data['EMAIL']

                self.save_prediction_results(email,prediction_label)

                chart_image = self.generate_chart()

                context = {
                    'prediction': prediction_Stroke,
                    'prediction_label': prediction_label,
                    'chart_image': chart_image
                }
                return render(request, 'Int_app/stroke_results.html', context)

            except Exception as e:
                logger.error(f"Prediction failed: {e}")
                return Response({'error': f'Prediction failed: {e}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        else:
            logger.error(f"Serializer errors: {serializer.errors}")
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def generate_chart(self):
        # Create a DataFrame from the predictions
        df = pd.DataFrame(self.__class__.predictions, columns=['date', 'prediction'])
        logger.debug(f"Prediction DataFrame: {df}")

        if df.empty:
            return None

        # Plot the data
        fig, ax = plt.subplots()
        ax.plot(df['date'], df['prediction'], marker='o', linestyle='-')
        ax.set_xlabel('Date')
        ax.set_ylabel('Prediction')
        ax.set_title('Stroke Predictions Over Time')

        # Format the x-axis to show dates in a vertical manner
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.xticks(rotation=90)

        # Save the plot to a BytesIO object and encode it as base64
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        chart_image = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)
        return chart_image

    @staticmethod
    def save_prediction_results(email, last_prediction):
        try:
            # Create a prediction entry
            prediction_entry = Prediction(
                disease='Stroke',
                prediction_value=last_prediction,
                timestamp=datetime.now()
            )

            # Check if a DiseasePrediction document already exists for the email
            disease_prediction = DiseasePrediction.objects(email=email).first()

            if disease_prediction:
                # Check if there is an existing prediction for depression on the same date
                today = datetime.now().date()
                existing_prediction = None
                for pred in disease_prediction.predictions:
                    if pred.disease == 'Stroke' and pred.timestamp.date() == today:
                        existing_prediction = pred
                        break

                if existing_prediction:
                    # Remove the existing prediction
                    disease_prediction.predictions.remove(existing_prediction)

                # Append the new prediction
                disease_prediction.predictions.append(prediction_entry)
                disease_prediction.save()
            else:
                # Create a new DiseasePrediction document
                new_disease_prediction = DiseasePrediction(
                    email=email,
                    predictions=[prediction_entry]
                )
                new_disease_prediction.save()

            logger.info("Prediction saved successfully.")
        except Exception as e:
            logger.error(f"Error saving prediction: {e}")


def prediction_Stroke_form(request):
    print("reaching..")
    return render(request, 'Int_app/Stroke.html')



# KIDNEY DISEASE

class PredictAndPlotKidneyDiseaseView(APIView):
    last_prediction = None
    predictions = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Load the model
        model_path = os.path.join(os.path.dirname(__file__), 'model', 'model_kidney_disease.pkl')
        encoder_path = os.path.join(os.path.dirname(__file__), 'model', 'model_kidney_disease_encoder.pkl')
        imputer1_path = os.path.join(os.path.dirname(__file__), 'model', 'model_kidney_disease_imputer1.pkl')
        imputer2_path = os.path.join(os.path.dirname(__file__), 'model', 'model_kidney_disease_imputer2.pkl')

        try:
            self.model = joblib.load(model_path)
            self.encoder = joblib.load(encoder_path)
            self.imputer1 = joblib.load(imputer1_path)
            self.imputer2 = joblib.load(imputer2_path)

            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = None
            self.encoder = None
            self.imputer1 = None
            self.imputer2 = None

    def post(self, request):
        logger.debug(f"Request data: {request.data}")
        serializer = PredictionSerializer_KidneyDisease(data=request.data)
        if serializer.is_valid():
            if self.model is None or self.imputer1 is None or self.encoder is None or self.imputer2 is None:
                logger.error("Model not loaded.")
                return Response({'error': 'Model not loaded'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            data = serializer.validated_data

            symptoms = {
                'age': data['AGE'],
                'bp': data['BP'],
                'sg': data['SG'],
                'al': data['AL'],
                'su': data['SU'],
                'rbc': data['RBC'],
                'pc': data['PC'],
                'pcc': data['PCC'],
                'ba': data['BA'],
                'bgr': data['BGR'],
                'bu': data['BU'],
                'sc': data['SC'],
                'sod': data['SOD'],
                'pot': data['POT'],
                'hemo': data['HEMO'],
                'htn': data['HTN'],
                'dm': data['DM'],
                'cad': data['CAD'],
                'appet': data['APPET'],
                'pe': data['PE'],
                'ane': data['ANE'],
            }
            symptoms_df = pd.DataFrame([symptoms])
            logger.debug(f"Symptoms DataFrame before imputation: {symptoms_df}")

            try:
                # Ensure the input has the same number of features expected by the imputers
                columns_imputer1 = ['age', 'bp', 'sg', 'al', 'su']
                columns_imputer2 = ['bgr', 'bu', 'sc', 'sod', 'pot', 'hemo']

                if len(columns_imputer1) == self.imputer1.statistics_.shape[0]:
                    symptoms_df[columns_imputer1] = self.imputer1.transform(symptoms_df[columns_imputer1])
                else:
                    raise ValueError(
                        f"Expected {len(columns_imputer1)} features for imputer1 but got {symptoms_df[columns_imputer1].shape[1]}")

                if len(columns_imputer2) == self.imputer2.statistics_.shape[0]:
                    symptoms_df[columns_imputer2] = self.imputer2.transform(symptoms_df[columns_imputer2])
                else:
                    raise ValueError(
                        f"Expected {len(columns_imputer2)} features for imputer2 but got {symptoms_df[columns_imputer2].shape[1]}")

                logger.debug(f"Symptoms DataFrame after imputation: {symptoms_df}")

                # Apply one-hot encoding
                symptoms_encoded = self.encoder.transform(symptoms_df)
                logger.debug(f"Symptoms after encoding: {symptoms_encoded}")

                # Convert to numpy array if necessary
                symptoms_np = symptoms_encoded.toarray() if hasattr(symptoms_encoded, 'toarray') else symptoms_encoded

                # Make prediction
                prediction_kidney_disease = self.model.predict(symptoms_np)[0]
                logger.debug(f"Prediction result: {prediction_kidney_disease}")
                self.__class__.last_prediction = prediction_kidney_disease
                self.__class__.predictions.append((datetime.now(), prediction_kidney_disease))
                prediction_label = 'YES' if prediction_kidney_disease >= 0.5 else 'NO'

                email = data['EMAIL']

                self.save_prediction_results(email, prediction_label)

                chart_image = self.generate_chart()

                context = {
                    'prediction': prediction_kidney_disease,
                    'prediction_label': prediction_label,
                    'chart_image': chart_image
                }
                return render(request, 'Int_app/kidney_results.html', context)

            except Exception as e:
                logger.error(f"Prediction failed: {e}")
                return Response({'error': f'Prediction failed: {e}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        else:
            logger.error(f"Serializer errors: {serializer.errors}")
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def generate_chart(self):
        # Create a DataFrame from the predictions
        df = pd.DataFrame(self.__class__.predictions, columns=['date', 'prediction'])
        logger.debug(f"Prediction DataFrame: {df}")

        if df.empty:
            return None

        # Plot the data
        fig, ax = plt.subplots()
        ax.plot(df['date'], df['prediction'], marker='o', linestyle='-')
        ax.set_xlabel('Date')
        ax.set_ylabel('Prediction')
        ax.set_title('Kidney Disease Predictions Over Time')

        # Format the x-axis to show dates in a vertical manner
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.xticks(rotation=90)

        # Save the plot to a BytesIO object and encode it as base64
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        chart_image = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)
        return chart_image

    @staticmethod
    def save_prediction_results(email, last_prediction):
        try:
            # Create a prediction entry
            prediction_entry = Prediction(
                disease='Kidney Disease',
                prediction_value=last_prediction,
                timestamp=datetime.now()
            )

            # Check if a DiseasePrediction document already exists for the email
            disease_prediction = DiseasePrediction.objects(email=email).first()

            if disease_prediction:
                # Check if there is an existing prediction for depression on the same date
                today = datetime.now().date()
                existing_prediction = None
                for pred in disease_prediction.predictions:
                    if pred.disease == 'Kidney Disease' and pred.timestamp.date() == today:
                        existing_prediction = pred
                        break

                if existing_prediction:
                    # Remove the existing prediction
                    disease_prediction.predictions.remove(existing_prediction)

                # Append the new prediction
                disease_prediction.predictions.append(prediction_entry)
                disease_prediction.save()
            else:
                # Create a new DiseasePrediction document
                new_disease_prediction = DiseasePrediction(
                    email=email,
                    predictions=[prediction_entry]
                )
                new_disease_prediction.save()

            logger.info("Prediction saved successfully.")
        except Exception as e:
            logger.error(f"Error saving prediction: {e}")

def prediction_KidneyDisease_form(request):
    print("reaching..")
    return render(request, 'Int_app/Kidney_Disease.html')



# PARKINSSON DISEASE


class PredictAndPlotParkinssonDiseaseView(APIView):
    last_prediction = None
    predictions = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Load the model and scaler
        model_path = os.path.join(os.path.dirname(__file__), 'model', 'model_parkinson.pkl')
        scaler_path = os.path.join(os.path.dirname(__file__), 'model', 'model_parkinson_scaler.pkl')

        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            logger.info("Model and scaler loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model or scaler: {e}")
            self.model = None
            self.scaler = None

    def post(self, request):
        logger.debug(f"Request data: {request.data}")
        serializer = PredictionSerializer_ParkinssonDisease(data=request.data)
        if serializer.is_valid():
            if self.model is None or self.scaler is None:
                logger.error("Model or scaler not loaded.")
                return Response({'error': 'Model or scaler not loaded'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            data = serializer.validated_data

            symptoms = {
                'MDVP_FO_HZ': [data['MDVP_FO_HZ']],
                'MDVP_FHI_HZ': [data['MDVP_FHI_HZ']],
                'MDVP_FLO_HZ': [data['MDVP_FLO_HZ']],
                'MDVP_JITTER_PERCENT': [data['MDVP_JITTER_PERCENT']],
                'MDVP_JITTER_ABS': [data['MDVP_JITTER_ABS']],
                'MDVP_RAP': [data['MDVP_RAP']],
                'MDVP_PPQ': [data['MDVP_PPQ']],
                'JITTER_DDP': [data['JITTER_DDP']],
                'MDVP_SHIMMER': [data['MDVP_SHIMMER']],
                'MDVP_SHIMMER_DB': [data['MDVP_SHIMMER_DB']],
                'SHIMMER_APQ3': [data['SHIMMER_APQ3']],
                'SHIMMER_APQ5': [data['SHIMMER_APQ5']],
                'MDVP_APQ': [data['MDVP_APQ']],
                'SHIMMER_DDA': [data['SHIMMER_DDA']],
                'NHR': [data['NHR']],
                'HNR': [data['HNR']],
                'PPE': [data['PPE']],
                'RPDE': [data['RPDE']],
                'DFA': [data['DFA']],
                'SPREAD1': [data['SPREAD1']],
                'SPREAD2': [data['SPREAD2']],
                'D2': [data['D2']],
            }

            symptoms_df = pd.DataFrame(symptoms)

            try:
                symptoms_scaled = self.scaler.transform(symptoms_df)
                symptoms_array = symptoms_scaled.toarray() if hasattr(symptoms_scaled, 'toarray') else symptoms_scaled
                prediction_parkinson_disease = self.model.predict(symptoms_array)[0]
                logger.debug(f"Prediction result: {prediction_parkinson_disease}")
                self.__class__.last_prediction = prediction_parkinson_disease
                self.__class__.predictions.append((datetime.now(), prediction_parkinson_disease))
                prediction_label = 'YES' if prediction_parkinson_disease >= 0.48 else 'NO'

                email = data['EMAIL']

                self.save_prediction_results(email, prediction_label)

                chart_image = self.generate_chart()

                context = {
                    'prediction': prediction_parkinson_disease,
                    'prediction_label': prediction_label,
                    'chart_image': chart_image
                }
                return render(request, 'Int_app/parkinsson_results.html', context)

            except Exception as e:
                logger.error(f"Prediction failed: {e}")
                return Response({'error': f'Prediction failed: {e}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        else:
            logger.error(f"Serializer errors: {serializer.errors}")
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def generate_chart(self):
        # Create a DataFrame from the predictions
        df = pd.DataFrame(self.__class__.predictions, columns=['date', 'prediction'])
        logger.debug(f"Prediction DataFrame: {df}")

        if df.empty:
            return None

        # Plot the data
        fig, ax = plt.subplots()
        ax.plot(df['date'], df['prediction'], marker='o', linestyle='-')
        ax.set_xlabel('Date')
        ax.set_ylabel('Prediction')
        ax.set_title('Parkinsson Disease Predictions Over Time')

        # Format the x-axis to show dates in a vertical manner
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.xticks(rotation=90)

        # Save the plot to a BytesIO object and encode it as base64
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        chart_image = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)
        return chart_image

    @staticmethod
    def save_prediction_results(email, last_prediction):
        try:
            # Create a prediction entry
            prediction_entry = Prediction(
                disease='Parkinsson Disease',
                prediction_value=last_prediction,
                timestamp=datetime.now()
            )

            # Check if a DiseasePrediction document already exists for the email
            disease_prediction = DiseasePrediction.objects(email=email).first()

            if disease_prediction:
                # Check if there is an existing prediction for depression on the same date
                today = datetime.now().date()
                existing_prediction = None
                for pred in disease_prediction.predictions:
                    if pred.disease == 'Parkinsson Disease' and pred.timestamp.date() == today:
                        existing_prediction = pred
                        break

                if existing_prediction:
                    # Remove the existing prediction
                    disease_prediction.predictions.remove(existing_prediction)

                # Append the new prediction
                disease_prediction.predictions.append(prediction_entry)
                disease_prediction.save()
            else:
                # Create a new DiseasePrediction document
                new_disease_prediction = DiseasePrediction(
                    email=email,
                    predictions=[prediction_entry]
                )
                new_disease_prediction.save()

            logger.info("Prediction saved successfully.")
        except Exception as e:
            logger.error(f"Error saving prediction: {e}")
def prediction_ParkinssonDisease_form(request):
    print("reaching..")
    return render(request, 'Int_app/Parkinsson_Disease.html')

def home(request):
    print("reaching..home")
    return render(request, 'Int_app/home.html')

def diseasepredicition(request):
    print("reaching intro...")
    return render(request, 'Int_app/DiseasePrediction.html')




# reading excel for medical tests
'''

excel_file_path = 'C:\\Users\\prani\\PycharmProjects\\Int_Project\\Int_ML_Project\\blood tests.xlsx'
sheet_name='Sheet1'

df=pd.read_excel(excel_file_path,sheet_name=sheet_name)

data=df.to_dict(orient='records')

for record in data:
    document=DiseaseTest_Coll(
        Condition=record['Condition'],
        Blood_Test=record['Blood Test']
    )
    document.save()


excel_file_path = 'C:\\Users\\prani\\PycharmProjects\\Int_Project\\Int_ML_Project\\medecine&disease_excel.xlsx'

df=pd.read_excel(excel_file_path)

df = df.drop_duplicates()

data=df.to_dict(orient='records')

for record in data:
    document=SymptomMedicine_Coll(
        Symptom=record['disease'],
        Medecine=record['drug']
    )
    document.save()'''

def add_symptom(request):
    if request.method == 'POST':
        form = SymptomForm(request.POST)
        if form.is_valid():
            symptom = form.cleaned_data['Symptom']

            # Query MongoDB using MongoEngine to find all matching entries
            medicines = SymptomMedicine.objects(Symptom=symptom)

            if medicines:
                context = {'medicines': medicines}  # Pass the list of medicines to the template
            else:
                context = {'error': 'No medication found for this symptom'}

            return render(request, 'Int_app/result.html', context)
    else:
        form = SymptomForm()

    return render(request, 'Int_app/add_symptoms.html', {'form': form})


def add_disease(request):
    if request.method == 'POST':
        form = DiseaseForm(request.POST)
        if form.is_valid():
            disease = form.cleaned_data['Disease']

            # Query MongoDB using MongoEngine to find all matching entries
            tests = MedicalTest.objects(Condition=disease)

            if tests:
                unique_tests = {test.Blood_Test for test in tests}
                context = {'tests': unique_tests}  # Pass the list of tests to the template
            else:
                context = {'error': 'No tests found for this disease'}

            return render(request, 'Int_app/result_test.html', context)
    else:
        form = DiseaseForm()

    return render(request, 'Int_app/add_disease.html', {'form': form})


#PREDICTION HISTORY

def fetch_medical_history(request):
    if request.method == 'POST':
        email = request.POST.get('email')  # Assuming the form field is named 'email'
        if email:
            # Query to fetch the medical history based on email
            medical_history = DiseasePrediction.objects(email=email).first()
            if medical_history:
                # Render template with fetched records
                return render(request, 'Int_app/medical_history.html', {'medical_history': medical_history})
            else:
                # Handle case where email doesn't match any records
                return render(request, 'Int_app/medical_history.html', {'error': 'No medical history found for this email.'})

    # Render initial form template for entering email
    return render(request, 'Int_app/fetch_medical_history.html')

#USER MANAGEMENT

class UserSignupView(APIView):

    def get(self, request):
        return render(request, 'Int_app/signup.html')

    def post(self, request):
        serializer = UserSignupSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return redirect('/home/')
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class UserLoginView(APIView):

    def get(self, request):
        return render(request, 'Int_app/login.html')


    def post(self, request):
        serializer = UserLoginSerializer(data=request.data)
        if serializer.is_valid():
            email = serializer.validated_data['email']
            password = serializer.validated_data['password']

            try:
                user = User.objects.get(email=email)
                if user.check_password(password):
                    return redirect('/home/')
                else:
                    return Response({"error": "Invalid credentials"}, status=status.HTTP_400_BAD_REQUEST)
            except User.DoesNotExist:
                return Response({"error": "User does not exist"}, status=status.HTTP_400_BAD_REQUEST)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class UserLogoutView(APIView):

    def get(self, request):
        # Log out the user
        logout(request)
        # Render the logout success template
        return render(request, 'Int_app/logout_success.html')



def main(request):
    print("reaching..")
    return render(request, 'Int_app/main_page.html')

