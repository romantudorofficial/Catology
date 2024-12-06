import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight



# Get the data.

def calculate_accuracy(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    correct_predictions = np.sum(y_true == y_pred)
    
    accuracy = correct_predictions / len(y_true)
    return accuracy

def load_data_from_csv(file_path):
    df = pd.read_csv(file_path)

    df.columns = df.columns.str.strip()
    columns_to_encode = [
        'Gender', 'Age', 'Breed', 'Number', 'Housing', 'Zone', 'Exterior Access',
        'Observations', 'Timide', 'Calme', 'Effrayé', 'Intelligent', 'Vigilant',
        'Perséverant', 'Affectueux', 'Amical', 'Solitaire', 'Brutal', 'Dominant',
        'Aggressive', 'Impulsive', 'Predictable', 'Distracted'
    ]

    for column in columns_to_encode:
        if column in df.columns:
            le = LabelEncoder()
            df[f'{column}_numeric'] = le.fit_transform(df[column].astype(str))

    target_column = 'Breed_numeric'

    if 'Breed' in df.columns:
        le = LabelEncoder()
        df[target_column] = le.fit_transform(df['Breed'].astype(str))

    feature_columns = [col for col in df.columns if col.endswith('_numeric') and col != target_column]
    X = df[feature_columns].values
    y = df[target_column].values.reshape(-1, 1)
    X = X.astype(float)
    X = (X - X.mean(axis = 0)) / X.std(axis = 0)

    return train_test_split(X, y, test_size=0.2, random_state=42)



file_path = '/Users/bili/Documents/GitHub/Catology/Project/Data/modified_data_for_training.csv'
X_train, X_test, y_train, y_test = load_data_from_csv(file_path)


class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train.flatten())
class_weights = dict(enumerate(class_weights))


input_size = X_train.shape[1]
hidden_size = 10
output_size = len(np.unique(y_train))
learning_rate = 0.0001
epochs = 5000


np.random.seed(42)
weights_input_hidden = np.random.randn(input_size, hidden_size) * np.sqrt(1 / input_size)
weights_hidden_output = np.random.randn(hidden_size, output_size) * np.sqrt(1 / hidden_size)
bias_hidden = np.zeros((1, hidden_size))
bias_output = np.zeros((1, output_size))



def sigmoid (x):
    return 1 / (1 + np.exp(-x))



def sigmoid_derivative (x):
    return x * (1 - x)



def softmax (x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)



def weighted_categorical_cross_entropy (y_true, y_pred):

    y_true_one_hot = np.eye(output_size)[y_true.flatten()]
    weights = np.array([class_weights[i] for i in y_true.flatten()]).reshape(-1, 1)
    loss = -np.sum(weights * y_true_one_hot * np.log(y_pred + 1e-9)) / y_true.shape[0]

    return loss



def forward_propagation (X):

    global hidden_layer_input, hidden_layer_output, final_output

    hidden_layer_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)
    final_output = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    final_output = softmax(final_output)

    return final_output



def backward_propagation (X, y, output):

    global weights_input_hidden, weights_hidden_output, bias_hidden, bias_output

    y_true_one_hot = np.eye(output_size)[y.flatten()]
    output_error = output - y_true_one_hot
    hidden_error = np.dot(output_error, weights_hidden_output.T)
    hidden_gradient = hidden_error * sigmoid_derivative(hidden_layer_output)
    weights_hidden_output -= np.dot(hidden_layer_output.T, output_error) * learning_rate
    bias_output -= np.sum(output_error, axis=0, keepdims=True) * learning_rate
    weights_input_hidden -= np.dot(X.T, hidden_gradient) * learning_rate
    bias_hidden -= np.sum(hidden_gradient, axis=0, keepdims=True) * learning_rate



errors = []

for epoch in range(epochs):

    predictions = forward_propagation(X_train)
    loss = weighted_categorical_cross_entropy(y_train, predictions)
    errors.append(loss)
    backward_propagation(X_train, y_train, predictions)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")



y_pred_test = forward_propagation(X_test)
y_pred_test = np.argmax(y_pred_test, axis=1)
y_test = y_test.flatten()

accuracy = calculate_accuracy(y_test, y_pred_test)
conf_matrix = confusion_matrix(y_test, y_pred_test)



# Display the results.

print("\nPerformance on the test dataset:")
print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)



# Display the graph.

plt.plot(errors)
plt.title("Convergence of Error During Training")
plt.xlabel("Epoch Number")
plt.ylabel("Error")
plt.show()