from django import forms
from .models import CustomUser
from django.contrib.auth.forms import AuthenticationForm


class UserRegistrationForm(forms.ModelForm):
    password = forms.CharField(widget=forms.PasswordInput)

    class Meta:
        model = CustomUser  # Change the model import to your custom user model
        fields = ['username', 'email', 'password', 'first_name', 'last_name', 'sex']


class UserLoginForm(AuthenticationForm):
    class Meta:
        model = CustomUser
        fields = ['username', 'password']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['username'].widget.attrs['class'] = 'form-control'
        self.fields['password'].widget.attrs['class'] = 'form-control'




