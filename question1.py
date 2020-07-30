import nltk
import pandas as pd
from keras.models import Model

nltk.download('punkt')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
import re
from keras.utils import to_categorical
import random
import tensorflow
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense,Dropout,Embedding,Conv1D,MaxPooling1D,Flatten, Input
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.models import Sequential
from tqdm import tqdm
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='bs4')
lemmatizer = WordNetLemmatizer()

# set random seed for the session and also for tensorflow that runs in background for keras
tensorflow.random.set_seed(123)
random.seed(123)

train= pd.read_csv("D:/UMKC/Subjects/Python/Exam2/question1/train.tsv", sep="\t")
test = pd.read_csv("D:/UMKC/Subjects/Python/Exam2/question1/test.tsv", sep="\t")

print(train.head())
print(test.head())
train.shape


def clean_sentences(df):
    reviews = []

    for sent in tqdm(df['Phrase']):
        # remove html content
        review_text = BeautifulSoup(sent).get_text()

        # remove non-alphabetic characters
        review_text = re.sub("[^a-zA-Z]", " ", review_text)

        # tokenize the sentences
        words = word_tokenize(review_text.lower())

        # lemmatize each word to its lemma
        lemma_words = [lemmatizer.lemmatize(i) for i in words]

        reviews.append(lemma_words)

    return (reviews)


# cleaned reviews for both train and test set retrieved
train_sentences = clean_sentences(train)
test_sentences = clean_sentences(test)
print('train_sentences length:', len(train_sentences))
print('test_sentences length:', len(test_sentences))

target=train.Sentiment.values
y_target=to_categorical(target)
print('y_target:', y_target)
num_classes=y_target.shape[1]
print('num_classes:', num_classes)


X_train,X_val,y_train,y_val=train_test_split(train_sentences,y_target,test_size=0.2,stratify=y_target)

# It is needed for initializing tokenizer of keras and subsequent padding

unique_words = set()
len_max = 0

for sent in tqdm(X_train):

    unique_words.update(sent)

    if (len_max < len(sent)):
        len_max = len(sent)

# length of the list of unique_words gives the no of unique words
print('length of unique words:', len(list(unique_words)))
print('Max length:', len_max)

tokenizer = Tokenizer(num_words=len(list(unique_words)))
tokenizer.fit_on_texts(list(X_train))
X_train = tokenizer.texts_to_sequences(X_train)
X_val = tokenizer.texts_to_sequences(X_val)
X_test = tokenizer.texts_to_sequences(test_sentences)

# padding done to equalize the lengths of all input reviews. CNN networks needs all inputs to be same length.
# Therefore reviews lesser than max length will be made equal using extra zeros at end. This is padding.
X_train = sequence.pad_sequences(X_train, maxlen=len_max)
X_val = sequence.pad_sequences(X_val, maxlen=len_max)
X_test = sequence.pad_sequences(X_test, maxlen=len_max)
print('All Shapes', X_train.shape,X_val.shape,X_test.shape)

# 1.a) Using CNN model and adding embedding layer

model=Sequential()
model.add(Embedding(len(list(unique_words)),300,input_length=len_max))
model.add(Conv1D(64,5,activation= 'tanh'))
model.add(Dropout(0.5))
model.add(Conv1D(64,5,activation= 'tanh'))
model.add(Dropout(0.5))
model.add(Dense(100,activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling1D(pool_size=4))
model.add(Flatten())
model.add(Dense(num_classes,activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.005), metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train, validation_data=(X_val, y_val),epochs=10, batch_size=256, verbose=1)
test_loss, test_acc = model.evaluate(X_val, y_val)
print(' After embedding test accuracy: ', test_acc)

# 1.b) Plot loss of the model.

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.legend(['Training Loss', 'Validation Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
