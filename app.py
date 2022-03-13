import pickle
import numpy as np
from flask import Flask, render_template, request

server = Flask(__name__)


@server.route('/', methods=['GET'])
def login():
    return render_template('index.html')


@server.route('/predict', methods=['POST'])
def predict_image():
    for each in request.files:
        print(each)
    image = request.files['file'].read()

    # plant_disease_model = open('plant_disease_model.pkl', 'rb')

    clf = pickle.load(open('plant_diesease_model.pkl', 'rb'))
    print("after open")

    # clf = pickle.load(plant_disease_model)
    print("after clf")
    # scores = clf.evaluate(image)
    # test_img = image.load_img(path, target_size=(48, 48))
    test_img = image.img_to_array(image)
    print("after image array")
    test_img = np.expand_dims(test_img, axis=0)
    print("after image expand")
    result = clf.predict(test_img)
    print("after image result")

    print(result)
    return render_template('index.html', prediction=result)


server.debug = True
if __name__ == '__main__':
    server.run(host='0.0.0.0', port=8080)
