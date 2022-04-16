from preset import sigmoid, softmax, x1, x2, needed_values
from classes import Net, flatten
import time


x1 = flatten(x1)
x2 = flatten(x2)

images = [x1, x2]

net = Net(images, sigmoid, needed_values, 0.01)
net.add_layer(3)
net.add_layer(2)
net.add_layer(2)
net.complete_net()
net.print_current_results()
print('начинаю обучение')
start_time = time.time()
for _ in range(10000):
    for i in range(2):
        net.set_image(i)
        net.calculate_deltas()
#        net.print()
        net.learn()
#        net.print()
net.print_current_results()
net.learn_scale = 0.1
for _ in range(10000):
    for i in range(2):
        net.set_image(i)
        net.calculate_deltas()
#        net.print()
        net.learn()
#        net.print()
print('обучение закончено')
print('время выполнения:', time.time()-start_time)
net.print_current_results()
net.save_weights('weights.txt')
