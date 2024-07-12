from mongoengine import Document, fields,EmbeddedDocument, EmbeddedDocumentField
from mongoengine import Document, StringField, ListField, DateTimeField,EmailField,DynamicField
import datetime
from datetime import datetime
from django.contrib.auth.hashers import make_password, check_password, make_password


class User(Document):
    username = StringField(required=True, unique=True)
    email = EmailField(required=True, unique=True)
    password = StringField(required=True)

    meta = {'collection': 'user_details'}

    def set_password(self, raw_password):
        self.password = make_password(raw_password)

    def check_password(self, raw_password):
        return check_password(raw_password, self.password)




class DiseaseTest_Coll(Document):
    Condition = fields.StringField(required=True, max_length=200)
    Blood_Test = fields.StringField(required=True, max_length=200)

    meta = {'collection': 'disease_test'}


class SymptomMedicine_Coll(Document):
    Symptom = fields.StringField(required=True, max_length=200)
    Medecine = fields.StringField(required=True, max_length=200)

    meta = {'collection': 'symptom_medecine'}





class Prediction(EmbeddedDocument):
    disease = StringField(required=True)
    prediction_value = DynamicField(required=True)
    timestamp = DateTimeField(default=datetime.now)

class DiseasePrediction(Document):
    email = StringField(required=True, unique=True)
    predictions = ListField(EmbeddedDocumentField(Prediction))

    meta = {'collection': 'disease_predictions'}




class SymptomMedicine(Document):
    Symptom = fields.StringField(required=True, max_length=200)
    Medecine = fields.StringField(required=True, max_length=200)

    meta = {'collection': 'symptom_medecine'}

class MedicalTest(Document):
    Condition = fields.StringField(required=True, max_length=200)
    Blood_Test = fields.StringField(required=True, max_length=200)

    meta = {
        'collection': 'disease_test'
    }


