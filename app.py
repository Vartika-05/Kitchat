from chatbot import get_response
from flask import Flask, render_template, request
import mysql.connector

app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg').lower()
    return str(get_response(userText))

if __name__ == "__main__":
    app.run() 