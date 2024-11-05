from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.authentication import SessionAuthentication, BasicAuthentication
from rest_framework.permissions import IsAuthenticated
from .prediction import Prediction
import json

# Create your views here.
@login_required
def index(request):
    return render(request, 'prediction_index.html')

@login_required
def lstm_prediction(req):
    return render(req, 'prediction_lstm.html')

class PredictionSVR(APIView):
    authentication_classes = [SessionAuthentication, BasicAuthentication]
    permission_classes = [IsAuthenticated]
    def get(self, request):
        df, mse_train, mse_test = Prediction.get_predictions()
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')


        json_string = df.to_json(orient='records')

        json_data = json.loads(json_string)
        response={
            'data':json_data,
            'mse_train':mse_train,
            'mse_test':mse_test
        }

        return Response(response)

class PredictionLSTM(APIView):
    authentication_classes = [SessionAuthentication, BasicAuthentication]
    permission_classes = [IsAuthenticated]
    def get(self, request):
        df, mse_train, mse_test = Prediction.get_lstm_predictions()
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')


        json_string = df.to_json(orient='records')

        json_data = json.loads(json_string)
        response={
            'data':json_data,
            'mse_train':mse_train,
            'mse_test':mse_test
        }

        return Response(response)
pass
