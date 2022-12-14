from flask import Flask, request, render_template
from keras.models import load_model
app = Flask('ML APP',template_folder='templates', static_folder='static')
model = load_model('SampleModel.h5')
Labels = ['Setosa', 'Versicolor', 'Virginica']


@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    features =[]
    for x in request.form.values():
        if (type(x) == int) or type(x) == float:
            features.append(float(x))
        else:
            return render_template('index.html', resultTXT=f'Oops! Input must be a real number!')
    prediction = model.predict(features)
    out = Labels[list(prediction[0]).index(prediction[0].max())]
    return render_template('index.html', resultTXT = f'Flower species is {out}.')

if __name__ == '__main__':
    app.run(debug=True)
