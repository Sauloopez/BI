from django.db import IntegrityError
from django.shortcuts import render, redirect
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.models import User
from django.contrib.auth import login, logout, authenticate
from .customForms import LoginForm, RegistryForm


# Create your views here.


def index(request):
    return render(request, 'home_index.html')


def signup(request):
    method = request.method
    if method == 'GET':
        return render(request, 'signup.html', {
            'form': RegistryForm
        })
    elif method == 'POST':
        form = RegistryForm(request.POST)
        if form.is_valid():
            try:
                form.clean()
                new_user = User.objects.create_user(username=form.cleaned_data['username'],
                                                    password=form.cleaned_data['password1'])
                new_user.save()
                return redirect('/accounts/login/')
            except IntegrityError:
                return render(request, 'signup.html', {
                    'form': RegistryForm,
                    'error': 'User Already Exists'
                })
        return render(request, 'signup.html', {
            'form': RegistryForm,
            'error': 'Passwords are not equals'
        })


def login_view(request):
    if request.method == 'GET':
        form = AuthenticationForm()
        return render(request, 'login.html', {
            'form': form
        })
    elif request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect(form.cleaned_data['next'])
        else:
            return render(request, 'login.html', {
                'form': form,
                'error' : 'Credenciales Inválidas'
            })
    pass