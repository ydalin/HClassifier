from flask import Flask, render_template, request
from main import get_joke


app = Flask(__name__)

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/script/')
def my_link():
  script_option = int(request.args.get("option"))
  if script_option in [1, 2, 3]:
    return get_joke(script_option)
  else:
    return "Incorrect selection"

if __name__ == '__main__':
  app.run(debug=True)
