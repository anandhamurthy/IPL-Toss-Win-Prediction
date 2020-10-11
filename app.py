from flask import Flask
import pickle
import numpy as np

filename = 'model.pkl'
classifier = pickle.load(open(filename, 'rb'))

result={'Sunrisers Hyderabad': 0,
 'Mumbai Indians': 1,
 'Royal Challengers Bangalore': 2,
 'Kolkata Knight Riders': 3,
 'Chennai Super Kings': 4,
 'Delhi Daredevils': 5,
 'Rajasthan Royals': 6,
 'Kings XI Punjab': 7}

app = Flask(__name__)

@app.route('/predict/<msg>',methods=['GET','POST'])
def predict(msg):
    l=msg.split(',')
    l = [[int(l[0]), int(l[1]), int(l[2]), int(l[3])]]
    l = np.array(l).reshape((1, -1))
    return list(result.keys())[list(result.values()).index(classifier.predict(l).tolist()[0])]


if __name__ == '__main__':
    app.run(debug=True)