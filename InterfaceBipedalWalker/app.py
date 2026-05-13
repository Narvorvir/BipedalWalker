from flask import Flask, render_template, request, redirect, url_for
from flask_socketio import SocketIO
import random
import time
import threading

app = Flask(__name__)
socketio = SocketIO(app)

models = {}
models['test'] = {'learningRate' : 'test'}

# Fonction qui simule l'évolution d'une donnée en arrière-plan
# def background_task():
#     money = 1500
#     while True:
#         time.sleep(2)
#         # On simule une variation
#         money += random.randint(-50, 100)
#         # On envoie la donnée au client via WebSocket
#         socketio.emit('update_data', {'money': money})

def error(error):
    return redirect(url_for('index') + "?error=" + error)

@app.route('/')
def index():
    error = request.args.get('error', '') 

    if error != None:
        print(error)

    return render_template('index.html')

@app.route('/build', methods=['POST'])
def build():
    result = request.form
    name = result['name']
    learningRate= result['learningRate']
    return render_template('build.html')

@app.route('/model', methods=['GET'])
def showModel():
    name = request.args.get('name', '')
    if name in models:
        return render_template('model.html', name = name)
    else:
        return error(f"Le model n'existe pas")

if __name__ == '__main__':
    # On lance la tâche de fond
    # threading.Thread(target=background_task, daemon=True).start()
    socketio.run(app, debug=True)