from flask import Flask, render_template, request
from modelo import treinar_modelo
import numpy as np
import requests

# Configuração
API_KEY = "c24b8d69a6974368b20639d82119270f"  # Insira sua chave de API aqui
app = Flask(__name__)

# Carregar modelo e histórico de pacientes
modelo, scaler, historico_df = treinar_modelo()

# Função para obter coordenadas a partir do CEP
def obter_coordenadas(cep):
    url = f"https://api.opencagedata.com/geocode/v1/json?q={cep}&key={API_KEY}&countrycode=BR"
    resposta = requests.get(url).json()
    if resposta["results"]:
        local = resposta["results"][0]["geometry"]
        return local["lat"], local["lng"]
    return None, None

@app.route("/", methods=["GET", "POST"])
def index():
    previsao = None
    prioridade = None

    if request.method == "POST":
        try:
            glicemia = float(request.form["glicemia"])
            temperatura = float(request.form["temperatura"])
            sintomas = int(request.form["sintomas"])
            cep = request.form["cep"]

            # Obter coordenadas do CEP
            latitude, longitude = obter_coordenadas(cep)
            if latitude is None or longitude is None:
                previsao = "Erro: CEP inválido ou não encontrado."
            else:
                # Prever gravidade
                entrada = np.array([[glicemia, temperatura, sintomas]])
                entrada_normalizada = scaler.transform(entrada)
                gravidade = modelo.predict(entrada_normalizada)[0]

                previsao = "Grave" if gravidade == 1 else "Leve"
                prioridade = "Alta" if gravidade == 1 else "Baixa"

                # Adicionar ao histórico
                novo_registro = {
                    "Glicemia": glicemia,
                    "Temperatura": temperatura,
                    "Sintomas": sintomas,
                    "Latitude": latitude,
                    "Longitude": longitude,
                    "Gravidade": previsao,
                    "Prioridade": prioridade
                }
                historico_df.loc[len(historico_df)] = novo_registro

        except ValueError:
            previsao = "Erro: Insira valores válidos."

    registros = historico_df.to_dict("records")
    return render_template("index.html", previsao=previsao, prioridade=prioridade, registros=registros)

if __name__ == "__main__":
    app.run(debug=True)
