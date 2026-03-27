import urllib.request
import json
import time

url = 'http://127.0.0.1:5000/predict'

# El mismo JSON que en el README
data = {
    "Country": "Afghanistan",
    "Year": 2015,
    "Status": "Developing",
    "Adult Mortality": 263.0,
    "infant deaths": 62,
    "Alcohol": 0.01,
    "percentage expenditure": 71.27,
    "Hepatitis B": 65.0,
    "Measles": 1154,
    "BMI": 19.1,
    "under-five deaths": 83,
    "Polio": 6.0,
    "Total expenditure": 8.16,
    "Diphtheria": 65.0,
    "HIV/AIDS": 0.1,
    "GDP": 584.25,
    "Population": 33736494.0,
    " thinness  1-19 years": 17.2,
    " thinness 5-9 years": 17.3,
    "Income composition of resources": 0.479,
    "Schooling": 10.1
}

req = urllib.request.Request(
    url,
    data=json.dumps(data).encode('utf-8'),
    headers={'Content-Type': 'application/json'}
)

print(f"Probando enviar datos a la API de Flask en {url} ...\n")
time.sleep(2) # Esperamos 2 segundos por si el servidor está subiendo

try:
    response = urllib.request.urlopen(req)
    result = json.loads(response.read().decode('utf-8'))
    print("Exito! El servidor de Flask respondio:")
    print(">>> Expectativa de Vida Predicha:", result['prediction'], "anos")
except Exception as e:
    print("Error al conectar con el servidor:", e)
