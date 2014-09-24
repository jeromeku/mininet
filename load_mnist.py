import cPickle
from mininet import load_data

datasets = load_data('mnist.pkl.gz')
train_set_x, train_set_y = datasets[0]
valid_set_x, valid_set_y = datasets[1]
test_set_x, test_set_y = datasets[2]

f = open('model.save', 'rb')
classifier = cPickle.load(f)

print((classifier.predict(test_set_x) != test_set_y).mean())
