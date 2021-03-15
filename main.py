import numpy as np
import matplotlib.pyplot as plt
data = open('communistmanifesto.txt', encoding = "utf8").read().lower()
characters = set(data)
number_of_char = len(characters)

char_to_index = {w: i for i,w in enumerate(characters)}
index_to_char = {i: w for i,w in enumerate(characters)}
class LSTM:
    def __init__(self, char_to_index, index_to_char, number_of_char, num_hid=128, sequence_length=25, 
                          epochs=10, lr=0.01, beta1=9e-1, beta2=999e-3):
        self.char_to_index = char_to_index 
        self.index_to_char = index_to_char 
        self.number_of_char = number_of_char 
        self.num_hid = num_hid 
        self.sequence_length = sequence_length
        self.epochs = epochs 
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
    
        #initialise weights and beta_inputs#
        self.params = {}
        std = (1.0/np.sqrt(self.number_of_char + self.num_hid)) # xavier initialization
        
        # forget gate
        self.params["weight_forget"] = np.random.randn(self.num_hid, self.num_hid + self.number_of_char) * std
        self.params["beta_forget"] = np.ones((self.num_hid,1))

        # input gate
        self.params["weight_input"] = np.random.randn(self.num_hid, self.num_hid + self.number_of_char) * std
        self.params["beta_input"] = np.zeros((self.num_hid,1))

        # cell gate
        self.params["weight_cell"] = np.random.randn(self.num_hid, self.num_hid + self.number_of_char) * std
        self.params["beta_cell"] = np.zeros((self.num_hid,1))

        # output gate
        self.params["weight_output"] = np.random.randn(self.num_hid, self.num_hid + self.number_of_char) * std
        self.params["beta_output"] = np.zeros((self.num_hid ,1))

        # output
        self.params["weighth_outputc"] = np.random.randn(self.number_of_char, self.num_hid) * \
                                          (1.0/np.sqrt(self.number_of_char))
        self.params["beta_outputc"] = np.zeros((self.number_of_char ,1))

        #initialise gradients and Adam parameters#
        self.grads = {}
        self.adam_params = {}

        for key in self.params:
            self.grads["d"+key] = np.zeros_like(self.params[key])
            self.adam_params["m"+key] = np.zeros_like(self.params[key])
            self.adam_params["v"+key] = np.zeros_like(self.params[key])
            
        self.smooth_loss = -np.log(1.0 / self.number_of_char) * self.sequence_length
        return
def sigmoid(self, x):
    return 1 / (1 + np.exp(-x))

LSTM.sigmoid = sigmoid


def softmax(self, x):
    e_x = np.exp(x - np.max(x)) 
    return e_x / np.sum(e_x)

LSTM.softmax = softmax
def clip_grads(self):
    for key in self.grads:
        np.clip(self.grads[key], -5, 5, out=self.grads[key])
    return

LSTM.clip_grads = clip_grads


def reset_grads(self):
    for key in self.grads:
        self.grads[key].fill(0)
    return

LSTM.reset_grads = reset_grads
def update_params(self, batch_num):
    for key in self.params:
        self.adam_params["m"+key] = self.adam_params["m"+key] * self.beta1 + \
                                    (1 - self.beta1) * self.grads["d"+key]
        self.adam_params["v"+key] = self.adam_params["v"+key] * self.beta2 + \
                                    (1 - self.beta2) * self.grads["d"+key]**2

        m_correlated = self.adam_params["m" + key] / (1 - self.beta1**batch_num)
        v_correlated = self.adam_params["v" + key] / (1 - self.beta2**batch_num) 
        self.params[key] -= self.lr * m_correlated / (np.sqrt(v_correlated) + 1e-8) 
    return

LSTM.update_params = update_params
def forward_step(self, x, h_prev, c_prev):
    z = np.row_stack((h_prev, x))

    f = self.sigmoid(np.dot(self.params["weight_forget"], z) + self.params["beta_forget"])
    i = self.sigmoid(np.dot(self.params["weight_input"], z) + self.params["beta_input"])
    c_bar = np.tanh(np.dot(self.params["weight_cell"], z) + self.params["beta_cell"])

    c = f * c_prev + i * c_bar
    o = self.sigmoid(np.dot(self.params["weight_output"], z) + self.params["beta_output"])
    h = o * np.tanh(c)

    v = np.dot(self.params["weighth_outputc"], h) + self.params["beta_outputc"]
    y_hat = self.softmax(v)
    return y_hat, v, h, o, c, c_bar, i, f, z

LSTM.forward_step = forward_step
def backward_step(self, y, y_hat, dh_next, dc_next, c_prev, z, f, i, c_bar, c, o, h):
    dv = np.copy(y_hat)
    dv[y] -= 1 # yhat - y

    self.grads["dweighth_outputc"] += np.dot(dv, h.T)
    self.grads["dbeta_outputc"] += dv

    dh = np.dot(self.params["weighth_outputc"].T, dv)
    dh += dh_next
    
    do = dh * np.tanh(c)
    da_o = do * o*(1-o)
    self.grads["dweight_output"] += np.dot(da_o, z.T)
    self.grads["dbeta_output"] += da_o

    dc = dh * o * (1-np.tanh(c)**2)
    dc += dc_next

    dc_bar = dc * i
    da_c = dc_bar * (1-c_bar**2)
    self.grads["dweight_cell"] += np.dot(da_c, z.T)
    self.grads["dbeta_cell"] += da_c

    di = dc * c_bar
    da_i = di * i*(1-i) 
    self.grads["dweight_input"] += np.dot(da_i, z.T)
    self.grads["dbeta_input"] += da_i

    df = dc * c_prev
    da_f = df * f*(1-f)
    self.grads["dweight_forget"] += np.dot(da_f, z.T)
    self.grads["dbeta_forget"] += da_f

    dz = (np.dot(self.params["weight_forget"].T, da_f)
         + np.dot(self.params["weight_input"].T, da_i)
         + np.dot(self.params["weight_cell"].T, da_c)
         + np.dot(self.params["weight_output"].T, da_o))

    dh_prev = dz[:self.num_hid, :]
    dc_prev = f * dc
    return dh_prev, dc_prev

LSTM.backward_step = backward_step
def forward_backward(self, x_batch, y_batch, h_prev, c_prev):
    x, z = {}, {}
    f, i, c_bar, c, o = {}, {}, {}, {}, {}
    y_hat, v, h = {}, {}, {}

    # Values at t= - 1
    h[-1] = h_prev
    c[-1] = c_prev

    loss = 0
    for t in range(self.sequence_length): 
        x[t] = np.zeros((self.number_of_char, 1))
        x[t][x_batch[t]] = 1

        y_hat[t], v[t], h[t], o[t], c[t], c_bar[t], i[t], f[t], z[t] = \
        self.forward_step(x[t], h[t-1], c[t-1])

        loss += -np.log(y_hat[t][y_batch[t],0])

    self.reset_grads()

    dh_next = np.zeros_like(h[0])
    dc_next = np.zeros_like(c[0])

    for t in reversed(range(self.sequence_length)):
        dh_next, dc_next = self.backward_step(y_batch[t], y_hat[t], dh_next, 
                                              dc_next, c[t-1], z[t], f[t], i[t], 
                                              c_bar[t], c[t], o[t], h[t]) 
    return loss, h[self.sequence_length-1], c[self.sequence_length-1]

LSTM.forward_backward = forward_backward
def sample(self, h_prev, c_prev, sample_size):
    x = np.zeros((self.number_of_char, 1))
    h = h_prev
    c = c_prev
    sample_string = "" 
    
    for t in range(sample_size):
        y_hat, _, h, _, c, _, _, _, _ = self.forward_step(x, h, c)        
        
        idx = np.random.choice(range(self.number_of_char), p=y_hat.ravel())
        x = np.zeros((self.number_of_char, 1))
        x[idx] = 1
        
        char = self.index_to_char[idx]
        sample_string += char
    return sample_string

LSTM.sample = sample
def train(self, X, verbeta_outputse=True):
    J = []  # to store losses

    num_batches = len(X) // self.sequence_length
    X_shortened = X[: num_batches * self.sequence_length] 

    for epoch in range(self.epochs):
        h_prev = np.zeros((self.num_hid, 1))
        c_prev = np.zeros((self.num_hid, 1))

        for j in range(0, len(X_shortened) - self.sequence_length, self.sequence_length):
            # prepare batches
            x_batch = [self.char_to_index[ch] for ch in X_shortened[j: j + self.sequence_length]]
            y_batch = [self.char_to_index[ch] for ch in X_shortened[j + 1: j + self.sequence_length + 1]]

            loss, h_prev, c_prev = self.forward_backward(x_batch, y_batch, h_prev, c_prev)

            # smooth out loss and store in list
            self.smooth_loss = self.smooth_loss * 0.999 + loss * 0.001
            J.append(self.smooth_loss)

            self.clip_grads()

            batch_num = epoch * self.epochs + j / self.sequence_length + 1
            self.update_params(batch_num)

            # print out loss and sample string
            if verbeta_outputse:
                if j % 400000 == 0:
                    print('Epoch:', epoch, '\tBatch:', j, "-", j + self.sequence_length,
                          '\tLoss:', round(self.smooth_loss, 2))
                    s = self.sample(h_prev, c_prev, sample_size=250)
                    print(s, "\n")
    return J, self.params

LSTM.train = train
model = LSTM(char_to_index, index_to_char, number_of_char, epochs = 10, lr = 0.01)

J, params = model.train(data)
plt.plot([i for i in range(len(J))], J)
plt.xlabel("#training iterations")
plt.ylabel("training loss")
plt.show()
