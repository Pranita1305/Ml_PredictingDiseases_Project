

from django.urls import path
from .views import PredictAndPlotLungCancerView, PredictAndPlotBreastCancerView, PredictAndPlotDepressionView, PredictAndPlotDiabetesView,PredictAndPlotHeartDiseaseView,PredictAndPlotSkinDiseaseView
from .views import prediction_LungCancer_form,prediction_BreastCancer_form,prediction_Depression_form, prediction_Diabetes_form,prediction_HeartDisease_form,prediction_SkinDisease_form

from .views import PredictAndPlotStrokeView
from .views import prediction_Stroke_form

from . import views
from .views import  PredictAndPlotParkinssonDiseaseView,PredictAndPlotKidneyDiseaseView,UserLogoutView
from .views import home,main,prediction_KidneyDisease_form,prediction_ParkinssonDisease_form
from .views import UserLoginView,UserSignupView,fetch_medical_history

from .views import diseasepredicition


urlpatterns = [
    path('prediction_LungCancer/', prediction_LungCancer_form, name='prediction_lung_cancer_form'),
    path('predict_LungCancer/', PredictAndPlotLungCancerView.as_view(), name='predict_lung_cancer'),
    #path('plot_LungCancer/', PlotPredictionsLungCancerView.as_view(), name='plot_predictions_lungcancer'),

    path('prediction_BreastCancer/', prediction_BreastCancer_form, name='prediction_breast_cancer_form'),
    path('predict_BreastCancer/', PredictAndPlotBreastCancerView.as_view(), name='predict_breast_cancer'),
    #path('plot_BreastCancer/', PlotPredictionsBreastCancerView.as_view(), name='plot_predictions_breastcancer'),

    path('prediction_Depression/', prediction_Depression_form, name='prediction_depression_form'),
    path('predict_Depression/', PredictAndPlotDepressionView.as_view(), name='predict_depression'),
    #path('plot_Depression/', PlotPredictionsDepressionView.as_view(), name='plot_predictions_depression'),

    path('prediction_Diabetes/', prediction_Diabetes_form, name='prediction_diabetes_form'),
    path('predict_Diabetes/', PredictAndPlotDiabetesView.as_view(), name='predict_diabetes'),
    #path('plot_Diabetes/', PlotPredictionsDiabetesView.as_view(), name='plot_predictions_diabetes'),

    path('prediction_HeartDisease/', prediction_HeartDisease_form, name='prediction_heart_disease_form'),
    path('predict_HeartDisease/', PredictAndPlotHeartDiseaseView.as_view(), name='predict_heart_disease'),
    #path('plot_HeartDisease/', PlotPredictionsHeartDiseaseView.as_view(), name='plot_predictions_heartdisease'),

    path('prediction_SkinDisease/', prediction_SkinDisease_form, name='prediction_skin_disease_form'),
    path('predict_SkinDisease/', PredictAndPlotSkinDiseaseView.as_view(), name='predict_skin_disease'),
    #path('plot_SkinDisease/', PlotPredictionsSkinDiseaseView.as_view(), name='plot_predictions_skindisease'),

    path('prediction_Stroke/', prediction_Stroke_form, name='prediction_stroke_form'),
    path('predict_Stroke/', PredictAndPlotStrokeView.as_view(), name='predict_stroke'),
    #path('plot_Stroke/', PlotPredictionsStrokeView.as_view(), name='plot_predictions_stroke'),

    path('prediction_KidneyDisease/', prediction_KidneyDisease_form, name='prediction_kidney_disease_form'),
    path('predict_KidneyDisease/', PredictAndPlotKidneyDiseaseView.as_view(), name='predict_kidney_disease'),
    #path('plot_KidneyDisease/', PlotPredictionsKidneyDiseaseView.as_view(), name='plot_predictions_kidneydisease'),

    path('prediction_ParkinssonDisease/', prediction_ParkinssonDisease_form, name='prediction_parkinsson_disease_form'),
    path('predict_ParkinssonDisease/', PredictAndPlotParkinssonDiseaseView.as_view(), name='predict_parkinsson_disease'),
    #path('plot_ParkinssonDisease/', PlotPredictionsParkinssonDiseaseView.as_view(), name='plot_predictions_parkinssondisease'),


    path('home/', home, name='home'),
    path('main/', main, name='main_page'),
    path('diseaseprediction/', diseasepredicition, name='disease_prediction'),
    path('medical_history/', fetch_medical_history, name='fetch_medical_history'),


    path('signup/', UserSignupView.as_view(), name='signup'),
    path('login/', UserLoginView.as_view(), name='login'),
    path('logout/', UserLogoutView.as_view(), name='logout'),




    path('add_symptom/', views.add_symptom, name='add_symptom'),
    path('add_disease/', views.add_disease, name='add_disease'),

    #path('prediction_history/',views.DiseasePrediction, name='pred_history'),
    #path('record_prediction/',views.record_prediction, name='record_prediction')
]

