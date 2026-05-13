from flask import Flask, render_template
from flask_socketio import SocketIO
import random
import time
import threading

app = Flask(__name__)
socketio = SocketIO(app)

# Fonction qui simule l'évolution d'une donnée en arrière-plan
def background_task():
    money = 1500
    while True:
        time.sleep(0.1)
        # On simule une variation
        money += random.randint(-50, 100)
        # On envoie la donnée au client via WebSocket
        socketio.emit('update_data', {'money': money})

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    # On lance la tâche de fond
    threading.Thread(target=background_task, daemon=True).start()
    socketio.run(app, debug=True)