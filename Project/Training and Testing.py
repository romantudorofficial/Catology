import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns



# Get the data.
file_path = 'Project\Data\modified_data_for_training_and_testing.csv'

# Verificăm dacă fișierul există
import os
if not os.path.exists(file_path):
    # Creează un fișier de exemplu pentru testare cu rase de pisici
    data = """Gender;Age;Breed;Number;Housing
Male;2;Siamese;1;Indoor
Female;3;Persian;2;Outdoor
Male;1;Maine Coon;1;Indoor
Female;4;Siamese;3;Outdoor
Male;3;Persian;2;Indoor
"""
    with open(file_path, 'w') as f:
        f.write(data)

# Încărcarea datelor
df = pd.read_csv(file_path, delimiter=';', na_values=['Nr', 'NSP'], encoding='utf-8')

# Tratarea valorilor lipsă (dacă există)
df = df.dropna()

# Codificarea variabilelor categorice
label_encoders = {}
for column in df.columns:
    if df[column].dtype == 'object':
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
        label_encoders[column] = le

# Împărțirea datelor în caracteristici (X) și țintă (y)
X = df.drop('Breed', axis=1).values
y = df['Breed'].values

# Împărțirea în seturi de antrenare și de testare
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Set de antrenare:", X_train.shape, y_train.shape)
print("Set de testare:", X_test.shape, y_test.shape)

# 2. Inițializarea parametrilor
input_size = X_train.shape[1]  # Numărul de caracteristici
hidden_size = 10               # Numărul de neuroni în stratul ascuns
output_size = len(np.unique(y))  # Numărul de clase
learning_rate = 0.01
num_epochs = 1000

# Inițializarea ponderilor și a bias-urilor
np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros((1, output_size))

# 3. Definirea funcțiilor de activare și a derivatelor

def relu(Z):
    return np.maximum(0, Z)

def relu_derivative(Z):
    return Z > 0

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return expZ / np.sum(expZ, axis=1, keepdims=True)

def cross_entropy_loss(Y_pred, Y_true):
    m = Y_true.shape[0]
    # Transformăm Y_true în format one-hot
    Y_true_one_hot = np.zeros_like(Y_pred)
    Y_true_one_hot[np.arange(m), Y_true] = 1
    # Calculăm pierderea
    loss = -np.sum(Y_true_one_hot * np.log(Y_pred + 1e-8)) / m
    return loss

def compute_accuracy(Y_pred, Y_true):
    predictions = np.argmax(Y_pred, axis=1)
    return accuracy_score(Y_true, predictions)

# 4. Propagarea înainte

def forward_propagation(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = softmax(Z2)
    cache = (Z1, A1, Z2, A2)
    return A2, cache

# 5. Propagarea înapoi

def backward_propagation(X, Y, cache, W2):
    m = X.shape[0]
    Z1, A1, Z2, A2 = cache
    # Transformăm Y în format one-hot
    Y_one_hot = np.zeros_like(A2)
    Y_one_hot[np.arange(m), Y] = 1

    # Calcularea gradientului pentru stratul de ieșire
    dZ2 = A2 - Y_one_hot
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    # Calcularea gradientului pentru stratul ascuns
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    return dW1, db1, dW2, db2

# 6. Antrenarea rețelei

# Pentru urmărirea pierderii și acurateței
losses = []
accuracies = []

for epoch in range(num_epochs):
    # Propagare înainte
    Y_pred, cache = forward_propagation(X_train, W1, b1, W2, b2)
    
    # Calcularea pierderii
    loss = cross_entropy_loss(Y_pred, y_train)
    losses.append(loss)
    
    # Calcularea acurateței
    if epoch % 100 == 0:
        acc = compute_accuracy(Y_pred, y_train)
        accuracies.append(acc)
        print(f"Epoca {epoch}: Pierdere = {loss:.4f}, Acuratețe = {acc:.4f}")
    
    # Propagare înapoi
    dW1, db1, dW2, db2 = backward_propagation(X_train, y_train, cache, W2)
    
    # Actualizarea ponderilor și a bias-urilor
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

# 7. Predicția pe setul de testare

def predict(X, W1, b1, W2, b2):
    Y_pred, _ = forward_propagation(X, W1, b1, W2, b2)
    predictions = np.argmax(Y_pred, axis=1)
    return predictions

# Realizarea predicțiilor
y_test_pred = predict(X_test, W1, b1, W2, b2)

# Evaluarea performanței
accuracy = accuracy_score(y_test, y_test_pred)
print(f"\nAcuratețe pe setul de testare: {accuracy:.4f}")

print("\nRaport de clasificare:")
print(classification_report(y_test, y_test_pred))

# Generarea matricei de confuzie
conf_matrix = confusion_matrix(y_test, y_test_pred)

# Obținerea denumirilor claselor
breed_le = label_encoders['Breed']
class_names = breed_le.classes_

# Crearea unui DataFrame pentru matricea de confuzie cu denumiri de clase
conf_matrix_df = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)

print("\nMatrice de confuzie:")
print(conf_matrix_df)

# Plotarea matricei de confuzie grafic
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_df, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Etichetă Predicată')
plt.ylabel('Etichetă Reală')
plt.title('Matrice de Confuzie')
plt.show()

# Generarea raportului de clasificare
class_report = classification_report(y_test, y_test_pred, output_dict=True)
class_report_df = pd.DataFrame(class_report).transpose()

print("\nRaport de clasificare:")
print(class_report_df)

# Plotarea raportului de clasificare grafic
plt.figure(figsize=(10, 6))
sns.heatmap(class_report_df.iloc[:-3, :-1], annot=True, cmap='YlGnBu')
plt.title('Raport de Clasificare')
plt.xlabel('Metrică')
plt.ylabel('Clasă')
plt.show()

# Plotarea evoluției pierderii și acurateței
plt.figure(figsize=(12, 5))

# Subplot pentru pierdere
plt.subplot(1, 2, 1)
plt.plot(range(num_epochs), losses, label='Pierdere')
plt.title('Evoluția Pierderii')
plt.xlabel('Epoci')
plt.ylabel('Pierdere')
plt.legend()

# Subplot pentru acuratețe
plt.subplot(1, 2, 2)
plt.plot(range(0, num_epochs, 100), accuracies, label='Acuratețe', color='green')
plt.title('Evoluția Acurateței')
plt.xlabel('Epoci')
plt.ylabel('Acuratețe')
plt.legend()

plt.tight_layout()
plt.show()