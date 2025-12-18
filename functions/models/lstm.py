import numpy as np

from functions.utils import sigmoid, dsigmoid, dtanh

# PREDIKSI
class lstm():
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        #bobot input
        self.bobot_i_fg = np.random.randn(hidden_size, input_size)
        self.bobot_i_ig = np.random.randn(hidden_size, input_size)
        self.bobot_i_cc = np.random.randn(hidden_size, input_size)
        self.bobot_i_og = np.random.randn(hidden_size, input_size)

        #bobot short-term (hidden)
        self.bobot_h_fg = np.random.randn(hidden_size, hidden_size)
        self.bobot_h_ig = np.random.randn(hidden_size, hidden_size)
        self.bobot_h_cc = np.random.randn(hidden_size, hidden_size)
        self.bobot_h_og = np.random.randn(hidden_size, hidden_size)

        #bias
        self.bias_fg = np.zeros((hidden_size, 1))
        self.bias_ig = np.zeros((hidden_size, 1))
        self.bias_cc = np.zeros((hidden_size, 1))
        self.bias_og = np.zeros((hidden_size, 1))

        # Output layer
        self.bobot_y = np.random.randn(output_size, hidden_size)
        self.bias_y = np.zeros((output_size, 1))

    def forward(self, input_sequence):
        #nilai untuk proses backpropagation
        self.cache = []

        prev_h = np.zeros((self.hidden_size, 1))
        prev_c = np.zeros((self.hidden_size, 1))

        output = []

        for i in range(len(input_sequence)):
            input = input_sequence[i].reshape(-1, 1)  

            fg = sigmoid(self.bobot_i_fg @ input + self.bobot_h_fg @ prev_h + self.bias_fg)
            ig = sigmoid(self.bobot_i_ig @ input + self.bobot_h_ig @ prev_h + self.bias_ig)
            cc = np.tanh(self.bobot_i_cc @ input + self.bobot_h_cc @ prev_h + self.bias_cc)
            og = sigmoid(self.bobot_i_og @ input + self.bobot_h_og @ prev_h + self.bias_og)

            #update cell state (long term)
            c = fg * prev_c + ig * cc

            #update hidden state (short term)
            h = og * np.tanh(c)

            #output
            y = self.bobot_y @ h + self.bias_y
            output.append(y)

            self.cache.append((input, prev_h, h, fg, ig, og, cc, prev_c, c))

        return output[-1]
    
    def backward(self, gradient_output):
        dWi_fg = np.zeros_like(self.bobot_i_fg)
        dWi_ig = np.zeros_like(self.bobot_i_ig)
        dWi_cc = np.zeros_like(self.bobot_i_cc)
        dWi_og = np.zeros_like(self.bobot_i_og)

        dWh_fg = np.zeros_like(self.bobot_h_fg)
        dWh_ig = np.zeros_like(self.bobot_h_ig)
        dWh_cc = np.zeros_like(self.bobot_h_cc)
        dWh_og = np.zeros_like(self.bobot_h_og)

        db_fg = np.zeros_like(self.bias_fg)
        db_ig = np.zeros_like(self.bias_ig)
        db_cc = np.zeros_like(self.bias_cc)
        db_og = np.zeros_like(self.bias_og)

        dWy = np.zeros_like(self.bobot_y)
        dby = np.zeros_like(self.bias_y)

        dy_final = gradient_output.reshape(-1, 1)
        dWy += dy_final @ self.cache[-1][2].T
        dby += dy_final

        dh_next = self.bobot_y.T @ dy_final
        dc_next = np.zeros((self.hidden_size, 1))

        #BPTT
        for i in reversed(range(len(self.cache))):
            x, h_prev, h, fg, ig, og, cc, c_prev, c = self.cache[i]

            dc = dh_next * og * dtanh(c) + dc_next

            dog = dh_next * np.tanh(c)
            dfg = dc * c_prev
            dig = dc * cc
            dcc = dc * ig

            og_input = self.bobot_i_og @ x + self.bobot_h_og @ h_prev + self.bias_og
            fg_input = self.bobot_i_fg @ x + self.bobot_h_fg @ h_prev + self.bias_fg
            ig_input = self.bobot_i_ig @ x + self.bobot_h_ig @ h_prev + self.bias_ig
            cc_input = self.bobot_i_cc @ x + self.bobot_h_cc @ h_prev + self.bias_cc

            dog_input = dog * dsigmoid(og_input)
            dfg_input = dfg * dsigmoid(fg_input)
            dig_input = dig * dsigmoid(ig_input)
            dcc_input = dcc * dtanh(cc_input)

            dWi_fg += dfg_input @ x.T
            dWi_ig += dig_input @ x.T
            dWi_cc += dcc_input @ x.T
            dWi_og += dog_input @ x.T

            dWh_fg += dfg_input @ h_prev.T
            dWh_ig += dig_input @ h_prev.T
            dWh_cc += dcc_input @ h_prev.T
            dWh_og += dog_input @ h_prev.T

            db_fg += dfg_input
            db_ig += dig_input
            db_cc += dcc_input
            db_og += dog_input

            dh_next = (self.bobot_h_fg.T @ dfg_input +
                       self.bobot_h_ig.T @ dig_input +
                       self.bobot_h_cc.T @ dcc_input +
                       self.bobot_h_og.T @ dog_input)

            dc_next = dc * fg

        #Update parameters
        self.bobot_i_fg -= self.learning_rate * dWi_fg
        self.bobot_i_ig -= self.learning_rate * dWi_ig
        self.bobot_i_cc -= self.learning_rate * dWi_cc
        self.bobot_i_og -= self.learning_rate * dWi_og

        self.bobot_h_fg -= self.learning_rate * dWh_fg
        self.bobot_h_ig -= self.learning_rate * dWh_ig
        self.bobot_h_cc -= self.learning_rate * dWh_cc
        self.bobot_h_og -= self.learning_rate * dWh_og

        self.bias_fg -= self.learning_rate * db_fg
        self.bias_ig -= self.learning_rate * db_ig
        self.bias_cc -= self.learning_rate * db_cc
        self.bias_og -= self.learning_rate * db_og

        self.bobot_y -= self.learning_rate * dWy
        self.bias_y -= self.learning_rate * dby

    def save_model(self, filepath):
        np.savez(
            filepath,

            bobot_i_fg=self.bobot_i_fg,
            bobot_i_ig=self.bobot_i_ig,
            bobot_i_cc=self.bobot_i_cc,
            bobot_i_og=self.bobot_i_og,

            bobot_h_fg=self.bobot_h_fg,
            bobot_h_ig=self.bobot_h_ig,
            bobot_h_cc=self.bobot_h_cc,
            bobot_h_og=self.bobot_h_og,

            bias_fg=self.bias_fg,
            bias_ig=self.bias_ig,
            bias_cc=self.bias_cc,
            bias_og=self.bias_og,

            # Output layer
            bobot_y=self.bobot_y,
            bias_y=self.bias_y,

            input_size=np.array([self.input_size]),
            hidden_size=np.array([self.hidden_size]),
            output_size=np.array([self.output_size])
        )

    def load_model(self, filepath):
        data = np.load(filepath)

        self.bobot_i_fg = data["bobot_i_fg"]
        self.bobot_i_ig = data["bobot_i_ig"]
        self.bobot_i_cc = data["bobot_i_cc"]
        self.bobot_i_og = data["bobot_i_og"]

        self.bobot_h_fg = data["bobot_h_fg"]
        self.bobot_h_ig = data["bobot_h_ig"]
        self.bobot_h_cc = data["bobot_h_cc"]
        self.bobot_h_og = data["bobot_h_og"]

        self.bias_fg = data["bias_fg"]
        self.bias_ig = data["bias_ig"]
        self.bias_cc = data["bias_cc"]
        self.bias_og = data["bias_og"]

        self.bobot_y = data["bobot_y"]
        self.bias_y = data["bias_y"]

# KLASIFIKASI
class lstm_klasifikasi():
    def __init__(self, vocab_size, dimensi_embedding, hidden_size, output_size, learning_rate):
        self.vocab_size = vocab_size
        self.dimensi_embedding = dimensi_embedding
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        #layer embedding [embed_dim, vocab_size]
        self.embedding = np.random.randn(dimensi_embedding, vocab_size) * np.sqrt(1.0 / dimensi_embedding)

        #bobot input
        self.bobot_i_fg = np.random.randn(hidden_size, dimensi_embedding)
        self.bobot_i_ig = np.random.randn(hidden_size, dimensi_embedding)
        self.bobot_i_cc = np.random.randn(hidden_size, dimensi_embedding)
        self.bobot_i_og = np.random.randn(hidden_size, dimensi_embedding)

        #bobot short-term (hidden)
        self.bobot_h_fg = np.random.randn(hidden_size, hidden_size)
        self.bobot_h_ig = np.random.randn(hidden_size, hidden_size)
        self.bobot_h_cc = np.random.randn(hidden_size, hidden_size)
        self.bobot_h_og = np.random.randn(hidden_size, hidden_size)

        #bias
        self.bias_fg = np.zeros((hidden_size, 1))
        self.bias_ig = np.zeros((hidden_size, 1))
        self.bias_cc = np.zeros((hidden_size, 1))
        self.bias_og = np.zeros((hidden_size, 1))

        # Output layer
        self.bobot_y = np.random.randn(output_size, hidden_size)
        self.bias_y = np.zeros((output_size, 1))

    def forward(self, input_sequence):
        #nilai untuk proses backpropagation
        self.cache = []

        prev_h = np.zeros((self.hidden_size, 1))
        prev_c = np.zeros((self.hidden_size, 1))

        for i in range(len(input_sequence)):
            word_id = int(input_sequence[i])  # e.g., 5

            #embedding lookup
            if word_id >= self.vocab_size:
                word_id = 1  # <UNK>
            input = self.embedding[:, word_id].reshape(-1, 1)  #(embed_dim, 1)

            fg = sigmoid(self.bobot_i_fg @ input + self.bobot_h_fg @ prev_h + self.bias_fg)
            ig = sigmoid(self.bobot_i_ig @ input + self.bobot_h_ig @ prev_h + self.bias_ig)
            cc = np.tanh(self.bobot_i_cc @ input + self.bobot_h_cc @ prev_h + self.bias_cc)
            og = sigmoid(self.bobot_i_og @ input + self.bobot_h_og @ prev_h + self.bias_og)

            #update cell state (long term)
            c = fg * prev_c + ig * cc

            #update hidden state (short term)
            h = og * np.tanh(c)

            self.cache.append((word_id, input, prev_h, h, fg, ig, og, cc, prev_c, c))
            prev_h, prev_c = h, c

        #output layer
        logits = self.bobot_y @ h + self.bias_y
        prob = sigmoid(logits)  # 0 - 1

        return prob
    
    def backward(self, gradient_output):
        dWi_fg = np.zeros_like(self.bobot_i_fg)
        dWi_ig = np.zeros_like(self.bobot_i_ig)
        dWi_cc = np.zeros_like(self.bobot_i_cc)
        dWi_og = np.zeros_like(self.bobot_i_og)

        dWh_fg = np.zeros_like(self.bobot_h_fg)
        dWh_ig = np.zeros_like(self.bobot_h_ig)
        dWh_cc = np.zeros_like(self.bobot_h_cc)
        dWh_og = np.zeros_like(self.bobot_h_og)

        db_fg = np.zeros_like(self.bias_fg)
        db_ig = np.zeros_like(self.bias_ig)
        db_cc = np.zeros_like(self.bias_cc)
        db_og = np.zeros_like(self.bias_og)

        dWy = np.zeros_like(self.bobot_y)
        dby = np.zeros_like(self.bias_y)

        d_embedding = np.zeros_like(self.embedding)

        dh_next = np.zeros((self.hidden_size, 1))
        dc_next = np.zeros((self.hidden_size, 1))

        #BPTT
        for i in reversed(range(len(gradient_output))):
            word_id, x, h_prev, h, fg, ig, og, cc, c_prev, c = self.cache[i]

            dy = gradient_output[i].reshape(-1, 1)
            dWy += dy @ h.T
            dby += dy

            dh = self.bobot_y.T @ dy + dh_next
            dc = dh * og * dtanh(c) + dc_next

            dog = dh * np.tanh(c)
            dfg = dc * c_prev
            dig = dc * cc
            dcc = dc * ig

            dog_input = dog * dsigmoid(self.bobot_i_og @ x + self.bobot_h_og @ h_prev + self.bias_og)
            dfg_input = dfg * dsigmoid(self.bobot_i_fg @ x + self.bobot_h_fg @ h_prev + self.bias_fg)
            dig_input = dig * dsigmoid(self.bobot_i_ig @ x + self.bobot_h_ig @ h_prev + self.bias_ig)
            dcc_input = dcc * dtanh(self.bobot_i_cc @ x + self.bobot_h_cc @ h_prev + self.bias_cc)

            dWi_fg += dfg_input @ x.T
            dWi_ig += dig_input @ x.T
            dWi_cc += dcc_input @ x.T
            dWi_og += dog_input @ x.T

            dWh_fg += dfg_input @ h_prev.T
            dWh_ig += dig_input @ h_prev.T
            dWh_cc += dcc_input @ h_prev.T
            dWh_og += dog_input @ h_prev.T

            db_fg += dfg_input
            db_ig += dig_input
            db_cc += dcc_input
            db_og += dog_input

            dx = (self.bobot_i_fg.T @ dfg_input +
                  self.bobot_i_ig.T @ dig_input +
                  self.bobot_i_cc.T @ dcc_input +
                  self.bobot_i_og.T @ dog_input)
            
            d_embedding[:, word_id] += dx.flatten()  # dx (embed_dim, 1)

            dh_next = (self.bobot_h_fg.T @ dfg_input +
                       self.bobot_h_ig.T @ dig_input +
                       self.bobot_h_cc.T @ dcc_input +
                       self.bobot_h_og.T @ dog_input)

            dc_next = dc * fg

        #Update parameters
        self.bobot_i_fg -= self.learning_rate * dWi_fg
        self.bobot_i_ig -= self.learning_rate * dWi_ig
        self.bobot_i_cc -= self.learning_rate * dWi_cc
        self.bobot_i_og -= self.learning_rate * dWi_og

        self.bobot_h_fg -= self.learning_rate * dWh_fg
        self.bobot_h_ig -= self.learning_rate * dWh_ig
        self.bobot_h_cc -= self.learning_rate * dWh_cc
        self.bobot_h_og -= self.learning_rate * dWh_og

        self.bias_fg -= self.learning_rate * db_fg
        self.bias_ig -= self.learning_rate * db_ig
        self.bias_cc -= self.learning_rate * db_cc
        self.bias_og -= self.learning_rate * db_og

        self.embedding -= self.learning_rate * d_embedding

        self.bobot_y -= self.learning_rate * dWy
        self.bias_y -= self.learning_rate * dby

    def save_model(self, filepath):
        np.savez(
            filepath,

            bobot_i_fg=self.bobot_i_fg,
            bobot_i_ig=self.bobot_i_ig,
            bobot_i_cc=self.bobot_i_cc,
            bobot_i_og=self.bobot_i_og,

            bobot_h_fg=self.bobot_h_fg,
            bobot_h_ig=self.bobot_h_ig,
            bobot_h_cc=self.bobot_h_cc,
            bobot_h_og=self.bobot_h_og,

            bias_fg=self.bias_fg,
            bias_ig=self.bias_ig,
            bias_cc=self.bias_cc,
            bias_og=self.bias_og,

            embedding=self.embedding,

            # Output layer
            bobot_y=self.bobot_y,
            bias_y=self.bias_y,

            vocab_size=np.array([self.vocab_size]),
            dimensi_embedding=np.array([self.dimensi_embedding]),
            hidden_size=np.array([self.hidden_size]),
            output_size=np.array([self.output_size])
        )

    def load_model(self, filepath):
        data = np.load(filepath)

        self.bobot_i_fg = data["bobot_i_fg"]
        self.bobot_i_ig = data["bobot_i_ig"]
        self.bobot_i_cc = data["bobot_i_cc"]
        self.bobot_i_og = data["bobot_i_og"]

        self.bobot_h_fg = data["bobot_h_fg"]
        self.bobot_h_ig = data["bobot_h_ig"]
        self.bobot_h_cc = data["bobot_h_cc"]
        self.bobot_h_og = data["bobot_h_og"]

        self.bias_fg = data["bias_fg"]
        self.bias_ig = data["bias_ig"]
        self.bias_cc = data["bias_cc"]
        self.bias_og = data["bias_og"]

        self.embedding = data["embedding"]

        self.bobot_y = data["bobot_y"]
        self.bias_y = data["bias_y"]