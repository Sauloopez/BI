from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.authentication import SessionAuthentication, BasicAuthentication
from rest_framework.permissions import IsAuthenticated
from .regression import Prediction
import json

# Create your views here.
@login_required
def index(request):
    return render(request, 'regression_index.html')

class PredictionSRV(APIView):
    # authentication_classes = [SessionAuthentication, BasicAuthentication]
    # permission_classes = [IsAuthenticated]
    def get(self, request):
        df = Prediction.get_predictions()
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')


        json_string = df.to_json(orient='records')

        json_data = json.loads(json_string)
        
        return Response(json_data)
pass