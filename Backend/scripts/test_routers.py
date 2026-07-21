
import io
import os
import sys
sys.path.insert(0, os.path.abspath('.'))

from PIL import Image
from fastapi.testclient import TestClient
from app.main import app

def create_dummy_image():
    img = Image.new('RGB', (100, 100), color='black')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    return img_bytes

def run_tests():
    print('Iniciando TestClient (cargando modelos...)')
    with TestClient(app) as client:
        print('? TestClient iniciado')
        
        # Test Root
        res = client.get('/')
        print(f'GET / -> {res.status_code}')
        
        # Test Health
        res = client.get('/health')
        print(f'GET /health -> {res.status_code} {res.json()}')
        
        dummy_img = create_dummy_image()
        
        # Test /api/v1/detection/vehicle
        res = client.post('/api/v1/detection/vehicle', files={'file': ('test.jpg', dummy_img.getvalue(), 'image/jpeg')})
        print(f'POST /api/v1/detection/vehicle -> {res.status_code}')
        
        # Test /api/v1/datasets
        res = client.get('/api/v1/datasets')
        print(f'GET /api/v1/datasets -> {res.status_code}')
        
        # Test /api/v1/anti-corruption/detections
        res = client.get('/api/v1/anti-corruption/detections')
        print(f'GET /api/v1/anti-corruption/detections -> {res.status_code}')
        
        # Multi detect (detect.py)
        dummy_img = create_dummy_image()
        res = client.post('/api/v1/detect', files={'file': ('test.jpg', dummy_img.getvalue(), 'image/jpeg')})
        print(f'POST /api/v1/detect -> {res.status_code}')
        
        print('? Pruebas completadas')

if __name__ == '__main__':
    run_tests()