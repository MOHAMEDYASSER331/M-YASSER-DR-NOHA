import numpy as np

class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(output_size, hidden_size) * 0.01
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))
        self.dWxh, self.dWhh, self.dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        self.dbh, self.dby = np.zeros_like(self.bh), np.zeros_like(self.by)

    def forward(self, inputs, h_prev):
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(h_prev)
        for t in range(len(inputs)):
            xs[t] = np.zeros((input_size, 1))
            xs[t][inputs[t]] = 1
            hs[t] = np.tanh(np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t-1]) + self.bh)
            ys[t] = np.dot(self.Why, hs[t]) + self.by
            ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
        return xs, hs, ps

    def backward(self, xs, hs, ps, targets):
        dh_next = np.zeros_like(hs[0])
        for t in reversed(range(len(xs))):
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1
            self.dWhy += np.dot(dy, hs[t].T)
            self.dby += dy
            dh = np.dot(self.Why.T, dy) + dh_next
            dh_raw = (1 - hs[t] * hs[t]) * dh
            self.dbh += dh_raw
            self.dWxh += np.dot(dh_raw, xs[t].T)
            self.dWhh += np.dot(dh_raw, hs[t-1].T)
            dh_next = np.dot(self.Whh.T, dh_raw)

    def update_weights(self, lr):
        for param, dparam in zip([self.Wxh, self.Whh, self.Why, self.bh, self.by],
                                [self.dWxh, self.dWhh, self.dWhy, self.dbh, self.dby]):
            param -= lr * dparam
        self.dWxh, self.dWhh, self.dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        self.dbh, self.dby = np.zeros_like(self.bh), np.zeros_like(self.by)

chars = list("helo")
data = [chars.index(c) for c in "hello"]
input_size = len(chars)
hidden_size = 10
output_size = len(chars)
lr = 0.1

rnn = SimpleRNN(input_size, hidden_size, output_size)
h_prev = np.zeros((hidden_size, 1))

for epoch in range(100):
    loss = 0
    xs, hs, ps = rnn.forward(data[:-1], h_prev)
    rnn.backward(xs, hs, ps, data[1:])
    rnn.update_weights(lr)
    for t in range(len(data)-1):
        loss += -np.log(ps[t][data[t+1]])
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")# M-YASSER-DR-NOHA
