from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import base64
from super_image import EdsrModel, ImageLoader
import torch

app = Flask(__name__)
CORS(app) # Permite que el HTML se comunique con Python sin bloqueos de seguridad

print("Cargando modelo de IA... (esto tomará unos segundos)")
modelo = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=4)
print("¡Modelo cargado y servidor listo!")

@app.route('/api/upscale', methods=['POST'])
def upscale_image():
    try:
        # 1. Recibir la imagen desde el HTML
        datos_json = request.get_json()
        image_data = datos_json['image'].split(',')[1]
        imagen_bytes = base64.b64decode(image_data)
        
        imagen_pil = Image.open(io.BytesIO(imagen_bytes)).convert("RGB")
        
        # 2. Procesar con Inteligencia Artificial (EDSR)
        inputs = ImageLoader.load_image(imagen_pil)
        preds = modelo(inputs)
        
        # 3. Guardar en memoria y codificar para devolver a la web
        output_buffer = io.BytesIO()
        ImageLoader.save_image(preds, output_buffer)
        output_buffer.seek(0)
        
        imagen_procesada_b64 = base64.b64encode(output_buffer.getvalue()).decode('utf-8')
        
        return jsonify({
            'status': 'success',
            'image': f'data:image/png;base64,{imagen_procesada_b64}'
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)