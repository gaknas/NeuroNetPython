import random
import copy
from preset import softmax

class Layer:
    def __init__(self, weights):
        self.weights = weights
        self.deltas = []
        self.before_softmax = 0
        for _ in self.weights[0]:
            self.deltas.append(0)

    def calculate(self, inp_values, activation_function):
        self.values = matmul(inp_values, self.weights)
        for i in range(len(self.values[0])):
            self.values[0][i] = activation_function(self.values[0][i])

    def calculate_deltas(self, next_weights, next_deltas):
        res = matmul(next_weights, transpond(next_deltas))
        for i in range(len(res)):
            self.deltas[i] = res[i][0]

    def correct_weights(self, learn_scale, previous_values):
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                self.weights[i][j] = self.weights[i][j] + learn_scale * self.deltas[j] * self.values[0][j] * (1 - self.values[0][j]) * previous_values[0][i]

    def __str__(self):
        return f'LAYER(values: {str(self.values)}; deltas: {str(self.deltas)}'
    
    def __repr__(self):
        return f'LAYER(values: {str(self.values)}; deltas: {str(self.deltas)}'

class InputLayer(Layer):
    def __init__(self, values):
        self.values = values
    def __str__(self):
        return str(self.values)
    def __repr__(self):
        return str(self.values)

class OutputLayer(Layer):
    def __init__(self, weights, values, deltas):
        super().__init__(weights)
        self.values = values
        self.deltas = deltas
    
    def calculate_deltas(self, need):
        for i in range(len(self.deltas)):
            self.deltas[i] = need[i] - self.values[0][i]

class Net:
    def __init__(self, images, activation_function, needed_values, learn_scale):
        for i in range(len(images)):
            images[i] = InputLayer(images[i])
        self.images = images
        self.layers = [images[0]]
        self.image = 0
        self.need = needed_values
        self.act = activation_function
        self.learn_scale = learn_scale

    def set_image(self, index):
        self.image = index
        self.layers[0] = self.images[index]
        self.calculate_layers()
    
    def add_layer(self, ln, weights=None):
        if weights:
            w = weights
        else:
            w = self.calculate_weights(len(self.layers[-1].values[0]), ln)
        lay = Layer(w)
        lay.calculate(self.layers[-1].values, self.act)
        self.layers.append(lay)

    def calculate_layer(self, index):
        lay = self.layers[index]
        """print()
        print(index)
        print('lay', lay)
        print('prlay', self.layers[index-1])
        print()"""
        lay.calculate(self.layers[index-1].values, self.act)

    def calculate_layers(self):
        for i in range(1, len(self.layers)):
            self.calculate_layer(i)

    def complete_net(self):
        data = self.layers[-1]
        self.layers[-1] = OutputLayer(data.weights, data.values, data.deltas)
    
    def calculate_weights(self, prelen, postlen):
        wgs = []
        for i in range(prelen):
            wgs.append(([0] * postlen).copy())
            for j in range(postlen):
                wgs[i][j] = random.random()
        return wgs

    def calculate_deltas(self):
        for i in range(len(self.layers) - 1, 0, -1):
            if i == (len(self.layers) - 1):
                self.layers[i].calculate_deltas(self.need[self.image])
            else:
                self.layers[i].calculate_deltas(self.layers[i + 1].weights, [self.layers[i + 1].deltas])

    def correct_weights(self):
        for i in range(1, len(self.layers)):
            self.layers[i].correct_weights(self.learn_scale, self.layers[i - 1].values)

    def learn(self):
        self.calculate_deltas()
        self.correct_weights()
        self.calculate_layers()

    def print(self):
        for i in range(len(self.layers)):
            print(self.layers[i])
            if i > 0:
                for i in self.layers[i].weights:
                    print(i)

    def last_layer(self):
        return self.layers[-1]

    def print_current_results(self):
        self.set_image(0)
        self.calculate_deltas()
        print('1 картинка: ', self.last_layer().values[0])
        self.set_image(1)
        self.calculate_deltas()
        print('2 картинка: ', self.last_layer().values[0])

    def load_weights(self, filename):
        file = open(filename, 'r')
        info = file.read().split('\n')
        file.close()
        for i in range(len(info)):
            info[i] = list(map(float, info[i].split(', ')[:-1]))
        res = [[]]
        iterat=0
        for i in range(len(info)):
            if info[i]:
                res[iterat].append(info[i])
            else:
                res.append([])
                iterat += 1
        while [] in res:
            res.remove([])
        for i in range(len(res)):
            res[i] = transpond(res[i])
        return res

    def save_weights(self, filename):
        file = open(filename, 'w')
        for i in self.layers[1:]:
            for j in i.weights:
                file.write(str(j))
                file.write('\n')
            file.write('\n')
        file.close()
#            if i > 0:
#                for j in self.layers[i].weights:
#                    print(j)

def matmul(mt1, mt2):
    if len(mt1[0]) != len(mt2):
        print(len(mt1[0]), len(mt2))
        return None
    else:
        nmt = []
        for i in range(len(mt1)):
            nmt.append(([0]*len(mt2[0])).copy())
            for j in range(len(mt2[0])):
                for k in range(len(mt1[0])):
                    nmt[i][j] += mt1[i][k] * mt2[k][j]
        return nmt

def flatten(mt):
    nmt = [[]]
    for i in mt:
        for j in i:
            nmt[0].append(j)
    return nmt

def transpond(mt):
    nmt = []
    for i in range(len(mt[0])):
        nmt.append(([0] * len(mt)).copy())
    for i in range(len(mt)):
        for j in range(len(mt[i])):
            nmt[j][i] = mt[i][j]
    return nmt
