import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample dataset
texts = [
    "I love this movie",
    "This film was amazing",
    "I really enjoyed the story",
    "I hate this movie",
    "This was the worst film",
    "Terrible acting and bad story"
]

labels = [1, 1, 1, 0, 0, 0]  # 1 = Positive, 0 = Negative

# Tokenization
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Padding
X = pad_sequences(sequences, maxlen=5)
y = np.array(labels)

# Model
model = Sequential([
    Embedding(input_dim=1000, output_dim=32, input_length=5),
    LSTM(32),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Train
model.fit(X, y, epochs=10, verbose=1)

# Test
test_text = ["I enjoyed the film"]
test_seq = tokenizer.texts_to_sequences(test_text)
test_pad = pad_sequences(test_seq, maxlen=5)

prediction = model.predict(test_pad)

print("\nPrediction:", "Positive" if prediction[0][0] > 0.5 else "Negative")
