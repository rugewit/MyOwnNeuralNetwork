

# Создание своего нейросетевого фреймворка

Задание: реализовать с нуля собственный нейросетевой фреймворк и решить задачи mnist, iris, mashroom.

#### Функции потерь

Было реализовано 3 функции потерь: MAE, MSE, CrossEntropyLoss. А также был создан абстрактный класс LossFunction для них. 

```python
class LossFunction(ABC):
    @abstractmethod
    def calc_diff(self, result_tensor, target_tensor):
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass


class MAE(LossFunction):
    def calc_diff(self, result_tensor, target_tensor):
        return np.mean(np.abs(target_tensor - result_tensor))

    def get_name(self) -> str:
        return "MAE"


class MSE(LossFunction):
    def calc_diff(self, result_tensor, target_tensor):
        return ((target_tensor - result_tensor)**2).mean()

    def get_name(self) -> str:
        return "MSE"
    

class CrossEntropyLoss(LossFunction):
    def calc_diff(self, result_tensor, target_tensor):
        epsilon = 1e-12
        predictions = np.clip(result_tensor, epsilon, 1. - epsilon)
        N = predictions.shape[0]
        ce = -np.sum(target_tensor*np.log(predictions+1e-9))/N
        return ce
    
    def get_name(self) -> str:
        return "CrossEntropyLoss"
```

#### Оптимизаторы

Было реализовано 4 оптимизатора: SGD, MomentumSGD, ClippingSGD, AdaptiveSGD, а также общий класс Optimizer для них. Код работы оптимизаторов будет приведен далее в методе back() у DenseLayer.

```python
class Optimizer:
    def __init__(self, learning_rate):
        self.lr = learning_rate


class SGD(Optimizer):
    def __init__(self, learning_rate):
        super().__init__(learning_rate)


class MomentumSGD(Optimizer):
    def __init__(self, learning_rate, momentum_rate):
        super().__init__(learning_rate)
        self.momentum_rate = momentum_rate


class ClippingSGD(Optimizer):
    def __init__(self, learning_rate, clip_threshold):
        super().__init__(learning_rate)
        self.clip_threshold = clip_threshold


class AdaptiveSGD(Optimizer):
    def __init__(self, learning_rate, beta1, beta2, delta):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.delta = delta
```

#### Слои

Абстрактный слой. Функция call для прямого прохода по нейросети, back для обратного распространения ошибки.

```python
class Layer(ABC):
    @abstractmethod
    def call(self, input_tensor):
        pass

    @abstractmethod
    def back(self, x, optim: Optimizer):
        pass
```

DenseLayer - слой с матрицами. В данном слое в методе back содержится код работы оптимизаторов.

```python
class DenseLayer(Layer):
    def __init__(self, input_nodes, out_nodes):
        self.input_nodes = input_nodes
        self.out_nodes = out_nodes
        # тензор (матрица)
        self.layer = np.random.normal(0.0, pow(self.input_nodes, -0.5), (self.out_nodes, self.input_nodes))
        # храним тензоры для обратного распространения ошибок
        self.prev_tensor = None
        self.res_tensor = None
        # for momentum
        self.velocity = np.zeros_like(self.layer)
        # for adaptive
        self.m = np.zeros_like(self.layer)
        self.v = np.zeros_like(self.layer)
    
    # вызываем при проходе по слоям
    def call(self, input_tensor):
        self.prev_tensor = input_tensor.copy()
        return np.dot(self.layer, input_tensor)
    
    # для обратного распространения ошибок
    def back(self, x, optim: Optimizer):
        lr = optim.lr
        gradient = np.dot((x * self.res_tensor * (1.0 - self.res_tensor)), np.transpose(self.prev_tensor))

        if type(optim) is SGD:
            self.layer += lr * gradient
            return np.dot(self.layer.T, x)
        elif type(optim) is MomentumSGD:
            self.velocity = -optim.momentum_rate * self.velocity + lr * gradient
            self.layer += self.velocity
            return np.dot(self.layer.T, x)
        elif type(optim) is ClippingSGD:
            clipped_gradient = np.clip(gradient, -optim.clip_threshold, optim.clip_threshold)
            self.layer += lr * clipped_gradient
            return np.dot(self.layer.T, x)
        elif type(optim) is AdaptiveSGD:
            self.m = optim.beta1 * self.m + (1 - optim.beta1) * gradient
            self.v = optim.beta2 * self.v + (1 - optim.beta2) * (gradient ** 2)
            self.layer += -1 * (-lr * self.m / (np.sqrt(self.v) + optim.delta))
            return np.dot(self.layer.T, x)
        else:
            raise Exception("Unknown optim")
```

#### Слои активации

Было реализовано 5 слоев активации: SigmoidLayer, TanhLayer, SoftPlusLayer, ReluLayer, SoftmaxLayer.

```python
class SigmoidLayer(Layer):
    def call(self, x):
        return 1 / (1 + np.exp(-x))
    
    def back(self, x, optim):
        return x


class TanhLayer(Layer):
    def call(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    
    def back(self, x, optim):
        return x


class SoftPlusLayer(Layer):
    def call(self, x):
        return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)
    
    def back(self, x, optim):
        return x
    

class ReluLayer(Layer):
    def call(self, x):
        return np.maximum(x, 0)
    
    def back(self, x, optim):
        return x


class SoftmaxLayer(Layer):
    def call(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x, axis=0)

    def back(self, x, optim):
        return x
```

#### Нейросеть

Далее представлен код нейросети. Она гибка в настройках: можно задать слои, выбрать скорость обучения, размер батча, задать количество эпох, выбрать функцию потерь. Метод start_train_batch позволяет запустить обучение, start_test позволяет запустить тест.

 ```python
 class NeuralNetwork:
     def __init__(self, layers):
         self.layers = layers
 
 
     # приватная функция для тренировки батча
     def _train_batch(self, inputs_list, targets_list, batch_size, features_size, classes_size, 
                     optim: Optimizer):
         inputs = np.array(inputs_list).reshape(batch_size, features_size).T
         targets = np.array(targets_list).reshape(batch_size, classes_size).T
 
         cur_tensor = inputs.copy()
         # идём по слоям
         for i in range(len(self.layers)):
             layer = self.layers[i]
             if isinstance(layer, (DenseLayer)):
                 cur_tensor = layer.call(cur_tensor)
             elif isinstance(layer, (SigmoidLayer, TanhLayer, SoftPlusLayer, ReluLayer, SoftmaxLayer)):
                 cur_tensor = layer.call(cur_tensor)
                 # сохраням для обратного распространения ошибок
                 self.layers[i-1].res_tensor = cur_tensor.copy()
             else:
                 raise Exception(f'Unknown class type: {type(layer)}')
 
         # тензор до обратного распространения ошибок
         actual_tensor = cur_tensor.copy()
         output_error = targets - actual_tensor
 
         cur_tensor = output_error.copy()
         # запускаем обратное распространение ошибок
         for layer in self.layers[::-1]:
             cur_tensor = layer.back(cur_tensor, optim)
         return (actual_tensor, targets)
 
 
     # запустить тренировку 
     def start_train_batch(self, all_inputs, all_targets, batch_size, features_size, classes_size, 
                           optim: Optimizer, epochs, loss_func, plot=True):
         # история по эпохам
         errors_history = []
 
         for _ in range(epochs):
             # перемешиваем индексы
             indices = np.random.permutation(len(all_inputs))
 
             epoch_error = 0.0
             for i in range(0, len(all_inputs), batch_size):
                 # формируем батчи
                 batch_indices = indices[i:i+batch_size]
                 batch_inputs_np = np.array([all_inputs[j] for j in batch_indices])
                 batch_targets_np = np.array([all_targets[j] for j in batch_indices])
                 result_tensor, target_tensor = self._train_batch(batch_inputs_np, batch_targets_np, batch_size, features_size, classes_size, optim)
                 # считаем loss
                 iter_error = loss_func.calc_diff(result_tensor, target_tensor)
                 epoch_error += iter_error
             # делим loss на кол-во итераций
             iterations = math.ceil(len(all_inputs) / batch_size)
             errors_history.append(epoch_error / iterations)
         
         # отображаем график
         if plot:
             plt.figure(figsize=(10, 6))
             plt.plot(list(range(1, epochs + 1)), errors_history)
 
             plt.title('Training Error History, method: ' + loss_func.get_name())
             plt.xlabel('Epoch')
             plt.ylabel('Error')
             plt.grid(True)
             plt.show()
             
     # получить 1 предсказание нейросети
     def query(self, inputs_list):
         inputs = np.array(inputs_list, ndmin=2).T
         
         cur_tensor = inputs.copy()
         # идем по слоям
         for layer in self.layers:
             cur_tensor = layer.call(cur_tensor)
     
         return cur_tensor
 
     # провести тестирование
     def start_test(self, input_test, target_test):
         scorecard = []
         for x, y in zip(input_test, target_test):
             correct_label = y
             outputs = self.query(x)
             label = np.argmax(outputs)
             if label == correct_label:
                 scorecard.append(1)
             else:
                 scorecard.append(0)
         scorecard_array = np.asarray(scorecard)
         res = scorecard_array.sum() / scorecard_array.size
         return res
 ```

#### Пример обучения MNIST

Для задачи MNIST задаются следующие начальные условия:

```python
features_size = 28 * 28
classes_size = 10
learning_rate = 0.1

layers = [DenseLayer(features_size, 200), SigmoidLayer(), DenseLayer(200, classes_size), SigmoidLayer()]
n = NeuralNetwork(layers)
optim = SGD(learning_rate)
loss_func = MAE()
```

После запуска тренировки, мы получаем такой график потерь по эпохам.

![1](https://github.com/MAILabs-Edu-2024/rugewit/MyOwnNeuralNetwork/blob/main/images/1.png)

**Итоговая точность: accuracy = 0.9736**

#### Пример обучения Iris

Начальные условия

```python
features_size = 4
classes_size = 3

learning_rate = 0.1

layers = [DenseLayer(features_size, 30), SigmoidLayer(), DenseLayer(30, classes_size), SigmoidLayer()]
n = NeuralNetwork(layers)
optim = SGD(learning_rate)
loss_func = MSE()
```

График потерь

![3](https://github.com/rugewit/MyOwnNeuralNetwork/blob/main/images/2.png)

**Итоговая точность: accuracy = 0.8**

#### Пример обучения Mushroom

Начальные условия

```python
features_size = 21
classes_size = 2

learning_rate = 0.1

layers = [DenseLayer(features_size, 200), SoftmaxLayer(), DenseLayer(200, classes_size), SigmoidLayer()]
n = NeuralNetwork(layers)
optim = MomentumSGD(learning_rate, 0.1)
loss_func = CrossEntropyLoss()
```

График потерь

![1](https://github.com/rugewit/MyOwnNeuralNetwork/blob/main/images/3.png)

**Итоговая точность: accuracy = 0.9778461538461538**
