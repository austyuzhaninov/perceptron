import numpy as np


# Функция сигмоид, для реализации активатора
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Входящие параметры, массив 4х3
training_inputs = np.array([
    [0, 0, 1],
    [1, 1, 1],
    [1, 0, 1],
    [0, 1, 1]
])

# Ожидаемые выходные данные (Т транспонирует матрицу превращая её 4х1)
training_outputs = np.array([[1, 1, 1, 1]]).T

np.random.seed(1)

# Получаем случайные веса, начальная точка обучения нейросети
synaptic_weights = 2 * np.random.random((3, 1)) - 1

print('Случайные инициализирующие веса (синапсы): ')
print(synaptic_weights)

# Метод обратного распространения
for i in range(2000):
    input_layer = training_inputs
    outputs = sigmoid(np.dot(input_layer, synaptic_weights))

    err = training_outputs - outputs
    adjustments = np.dot(input_layer.T, err * (outputs * (1 - outputs)))

    synaptic_weights += adjustments

print('Веса после обучения: ')
print(synaptic_weights)
print('Рузультат после обучения: ')
print(outputs)

# Новая ситуация
new_inputs = np.array([
    [1, 0, 1],
    [1, 1, 0],
    [1, 0, 0],
    [0, 1, 1]
])
outputs = sigmoid(np.dot(new_inputs, synaptic_weights))
print('Новая ситуация на основе полученного опыта: ')
print(outputs)
