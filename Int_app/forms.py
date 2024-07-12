from django import forms

class SymptomForm(forms.Form):
    Symptom = forms.CharField(label='Symptom', max_length=100)



class DiseaseForm(forms.Form):
    Disease = forms.CharField(label='Disease', max_length=100)


