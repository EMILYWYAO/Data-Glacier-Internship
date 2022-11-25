from flask import Flask, request, render_template
from keras.models import load_model
app = Flask('ML APP',template_folder='Week 4/templates', static_folder='Week 4/static')
model = load_model('Week 4/SampleModel.h5')
Labels = ['Setosa', 'Versicolor', 'Virginica']


@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    features = [[float(x) for x in request.form.values()]]
    print(features)
    prediction = model.predict(features)
    out = Labels[list(prediction[0]).index(prediction[0].max())]
    return render_template('index.html', resultTXT = f'Flower species is {out}.')

if __name__ == '__main__':
    app.run(debug=True)