from flask import Flask, render_template, request, redirect, url_for
from flask_socketio import SocketIO

app = Flask(__name__)

socketio = SocketIO(app)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        room_id = request.form['room_id']
        return redirect(url_for("enter_room", room_id = room_id))
    return render_template("home.html")

@app.route("/room/<string:room_id>", methods = ["GET", "POST"])
def enter_room(room_id):
    return render_template("chatroom.html", room_id=room_id)


if __name__ == "__main__":
    socketio.run(app, debug=True)