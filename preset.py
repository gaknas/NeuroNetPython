import math
formula1 = lambda x,y: 0.01 * 13 * (y + 1) * pow(-1, (x + 1))
formula2 = lambda x,y: 0.01 * 13 * (x + 1) * pow(-1, (x + 1))
sigmoid = lambda x: 1/(1+math.exp(-x))
x1 = [[0,1,1,0], [0,1,1,0], [0,0,0,0], [0,0,0,0]]
x2 = [[0,0,0,0], [0,0,0,0], [1,1,0,0], [1,1,0,0]]
needed_values = [[1,0], [0,1]]

def softmax(mt):
    sm = 0
    for i in mt:
        sm += math.exp(i)
    for i in range(len(mt)):
        mt[i] = math.exp(mt[i])/sm
    return mt

