from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Import IMDB Dataset
training_set, testing_set = imdb.load_data(num_words = 10000)
X_train, y_train = training_set
X_test, y_test = testing_set

print(f"Number of training samples = {X_train.shape[0]}")
print(f"Number of testing samples = {X_test.shape[0]}")

X_train_padded = sequence.pad_sequences(X_train, maxlen = 100)
X_test_padded = sequence.pad_sequences(X_test, maxlen = 100)

print(f"X_train vector shape = {X_train_padded.shape}")
print(f"X_test vector shape = {X_test_padded.shape}")

# Model Building 

def train_model(Optimizer, X_train, y_train, X_val, y_val):
	model = Sequential()
	model.add(Embedding(input_dim = 10000, output_dim = 128))
	model.add(LSTM(units=128))
	model.add(Dense(units = 1, activation = 'sigmoid'))
	model.compile(loss = 'binary_crossentropy', optimizer = Optimizer, metrics = ['accuracy'])
	scores = model.fit(X_train, y_train, batch_size = 128, epochs = 10, validation_data = (X_val, y_val))
	return scores, model


# Train model
RMSprop_score, RMSprop_model = train_model(Optimizer = 'RMSprop', X_train = X_train_padded, y_train = y_train,
	X_val = X_test_padded, y_val = y_test)

# Plot accuracy per epoch
plt.plot(range(1,11), RMSprop_score.history['accuracy'], label='Training Accuracy')
plt.axis([1,10,0,1])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Train and Validation Accuracy using RMSProp Optimizer')
plt.legend()
plt.show()


y_test_pred = RMSprop_model.predict_classes(X_test_padded)
c_matrix = confusion_matrix(y_test, y_test_pred)
ax = sns.heatmap(c_matrix, annot=True, xticklabels=['Negative Sentiment', 'Positive Sentiment'], yticklabels = ['Negative Sentiment', 'Positive Sentiment'], cbar = False, cmap = 'Blues', fmt = 'g')
ax.set_xlabel('Prediction')
ax.set_ylabel('Actual')
plt.show()