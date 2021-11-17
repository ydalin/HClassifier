from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/script/')
def my_link():
  script_option = int(request.args.get("option"))

  if script_option == 1:
    return "1"
  elif script_option == 2:
    return "2"
  elif script_option == 3:
    return "3"

  return "Incorrect selection"

if __name__ == '__main__':
  app.run(debug=True)
