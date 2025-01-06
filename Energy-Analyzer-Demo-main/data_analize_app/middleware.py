from django.shortcuts import redirect
from django.urls import reverse
from django.conf import settings

class PasswordProtectionMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response
        # Define your password here or in settings.py
        self.PASSWORD = "energy123"

    def __call__(self, request):
        # Exclude the login page from password protection
        if request.path == reverse('login'):
            return self.get_response(request)

        # Check if user is authenticated
        if not request.session.get('is_authenticated', False):
            return redirect('login')

        response = self.get_response(request)
        return response