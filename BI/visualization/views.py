from django.http import JsonResponse
from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.pagination import PageNumberPagination
import requests
from rest_framework.response import Response

class NASADataView(APIView):
    pagination_class = PageNumberPagination

    def get(self, request):
        url = ("""https://power.larc.nasa.gov/api/temporal/daily/point?parameters=T2M,T2MDEW,T2MWET,TS,T2M_RANGE,T2M_MAX,T2M_MIN&community=RE&longitude=-73.62664&latitude=4.142&start=20200101&end=20230630&format=json""")
        response = requests.get(url)
        data = response.json()

        t2m_values = data.get('properties', {}).get('parameter', {}).get('T2M', {})
        t2max_data = data.get('properties', {}).get('parameter', {}).get('T2M_MAX', {})
        t2min_data = data.get('properties', {}).get('parameter', {}).get('T2M_MIN', {})
        t2range_data = data.get('properties', {}).get('parameter', {}).get('T2M_RANGE', {})

        # Número de elementos por página
        items_per_page = 40

        # Divide los datos en páginas
        pages = []
        current_page = []
        count_pages = 0
        count_data=0

        for date in t2m_values.keys():
            count_data+=1
            row = {
                'nro':count_data,
                'date': date,
                't2m': t2m_values[date],
                't2max': t2max_data[date],
                't2min': t2min_data[date],
                't2range': t2range_data[date],
            }
            current_page.append(row)
            count_pages += 1

            if count_pages == items_per_page:
                pages.append(current_page)
                current_page = []
                count_pages = 0

        if current_page:
            pages.append(current_page)

        # Devuelve todas las páginas enumeradas en formato JSON
        return Response(pages)



def index(request):
    return render(request, 'visualization_index.html')