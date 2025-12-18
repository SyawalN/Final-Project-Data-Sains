import numpy as np
import joblib

from functions.utils import sigmoid

# PREDICTION
class backpropagation:
    def __init__(self, learning_rate=0.1, input=1):
        self.bobot = np.array(np.random.randn(input))
        self.bias = np.random.randn()
        self.learning_rate = learning_rate
    
    def mse(self, pred, target):
        return (pred - target)**2
    
    def prediksi(self, input):
        dot = np.dot(input, self.bobot) + self.bias
        aktivasi = sigmoid(dot)

        return aktivasi
    
    def gradient(self, input, target):
        dot = np.dot(input, self.bobot) + self.bias
        aktivasi = sigmoid(dot)

        turunan_mse = 2 * (aktivasi - target)
        turunan_sigmoid = aktivasi * (1 - aktivasi)
        turunan_bias = 1
        turunan_bobot = input

        gradient_bias = (turunan_mse * turunan_sigmoid * turunan_bias)
        gradient_bobot = (turunan_mse * turunan_sigmoid * turunan_bobot)

        return gradient_bias, gradient_bobot
    
    def update_parameter(self, gradient_bias, gradient_bobot):
        self.bias -= gradient_bias * self.learning_rate
        self.bobot -= gradient_bobot * self.learning_rate

    def train(self, input, target, steps):
        error_per_steps = []
        for i in range(steps):
            random_index = np.random.randint(len(input))

            input_index = input[random_index]
            target_index = target[random_index]

            gradient_bias, gradient_bobot = self.gradient(input_index, target_index)
            self.update_parameter(gradient_bias, gradient_bobot)

            if i % 100 == 0:
                jumlah_error = 0
                for j in range(len(input)):
                    pred = self.prediksi(input[j])
                    error = self.mse(pred, target[j])
                    jumlah_error += error
                
                error_per_steps.append(jumlah_error / len(input))

        return error_per_steps
    
    def validasi(self, input, target, scl):
        scaler = joblib.load(scl)
        output = np.zeros(len(input))

        total_error = 0
        for i in range(len(input)):
            pred = self.prediksi(input[i])
            error = self.mse(pred, target[i])
            total_error += error
            output[i] = scaler.inverse_transform([[pred]])[0][0]

        rata_rata = total_error / len(input)

        return output, scaler, rata_rata
    
    def save_model(self, file):
        np.savez(file, bobot = self.bobot, bias = self.bias)

    def load_model(self, file):
        parameter = np.load(file)
        self.bobot = parameter["bobot"]
        self.bias = parameter["bias"]

# CLASSIFICATION
class backpropagation_klasifikasi():
    def __init__(self, learning_rate=0.1, input=1):
        self.bobot = np.random.randn(input)
        self.bias = np.random.randn()
        self.learning_rate = learning_rate
    
    def cross_entropy(self, pred, target):
        pred = np.clip(pred, 1e-7, 1 - 1e-7)
        return - (target * np.log(pred) + (1 - target) * np.log(1 - pred))
    
    def prediksi(self, input):
        dot = np.dot(input, self.bobot) + self.bias
        aktivasi = sigmoid(dot)

        return aktivasi
    
    def gradient(self, input, target):
        dot = np.dot(input, self.bobot) + self.bias
        aktivasi = sigmoid(dot)

        turunan = aktivasi - target

        gradient_bias = turunan
        gradient_bobot = turunan * input

        return gradient_bias, gradient_bobot
    
    def update_parameter(self, gradient_bias, gradient_bobot):
        self.bias -= gradient_bias * self.learning_rate
        self.bobot -= gradient_bobot * self.learning_rate

    def train(self, input, target, steps):
        error_per_steps = []
        for i in range(steps):
            random_index = np.random.randint(len(input))

            input_index = input[random_index]
            target_index = target[random_index]

            gradient_bias, gradient_bobot = self.gradient(input_index, target_index)
            self.update_parameter(gradient_bias, gradient_bobot)

            if i % 100 == 0:
                jumlah_error = 0
                for j in range(len(input)):
                    pred = self.prediksi(input[j])
                    error = self.cross_entropy(pred, target[j])
                    jumlah_error += error
                
                error_per_steps.append(jumlah_error / len(input))

        return error_per_steps
    
    def validasi(self, input, target):
        output = []
        benar = 0
        total_error = 0

        for i in range(len(input)):
            pred_prob = self.prediksi(input[i])
            loss = self.cross_entropy(pred_prob, target[i])
            total_error += loss

            pred_label = 1 if pred_prob >= 0.5 else 0
            output.append(pred_label)

            if pred_label == target[i]:
                benar += 1

        akurasi = benar / len(input)
        rata_rata = total_error / len(input)

        return np.array(output), akurasi, rata_rata
    
    def save_model(self, file):
        np.savez(file, bobot = self.bobot, bias = self.bias)

    def load_model(self, file):
        parameter = np.load(file)
        self.bobot = parameter["bobot"]
        self.bias = parameter["bias"]