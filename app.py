import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__, template_folder="templates")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analizar", methods=["POST"])
def analizar():
    if "imagen" not in request.files:
        return jsonify({"error": "No se envió ninguna imagen"}), 400

    archivo = request.files["imagen"]
    npimg = np.frombuffer(archivo.read(), np.uint8)
    imagen = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if imagen is None:
        return jsonify({"error": "No se pudo decodificar la imagen"}), 400

    alto, ancho, canales = imagen.shape
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    nitidez = float(cv2.Laplacian(gris, cv2.CV_64F).var())

    try:
        f = np.fft.fft2(gris)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
        anomalia_fft = float(np.mean(magnitude_spectrum))
    except Exception:
        anomalia_fft = 0.0

    alerta_ia = anomalia_fft > 600 or nitidez < 50

    resultado = {
        "estado": "exito",
        "dimensiones": f"{ancho}x{alto}",
        "nitidez": round(nitidez, 2),
        "indice_fft": round(anomalia_fft, 2),
        "veredicto_preliminar": "Imagen generada por IA 🤖" if alerta_ia else "Foto Real (Cámara) 📸"
    }

    return jsonify(resultado)

if __name__ == "__main__":
    app.run(debug=True)
