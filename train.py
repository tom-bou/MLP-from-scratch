from model import Model
from process import load_dataset
import numpy as np

(train_x, train_y), (test_x, test_y) = load_dataset()

input_dim = train_x.shape[1]
hidden_dim = 100
output_dim = 10
learning_rate = 0.1
print("Creating model...")
model = Model(input_dim, hidden_dim, output_dim, learning_rate)
epoch = 20
batch_size = 64
print("Training model...")

for i in range(epoch):
    for j in range(0, train_x.shape[0], batch_size):
        x = train_x[j:j+batch_size]
        y = train_y[j:j+batch_size]
        
        a1, a2 = model.forward_pass(x)
        loss = model.compute_loss(y, a2)
        model.backward_pass(x, y, a2)
        
        print(f"Epoch: {i}, Loss: {loss}")

    
# Test and accuracy
_, a2 = model.forward_pass(test_x)
test_loss = model.compute_loss(test_y, a2)
accuracy = np.mean(np.argmax(a2, axis=1) == test_y)
print(f"Accuracy: {accuracy}")

print(f"Test Loss: {test_loss}")
    

