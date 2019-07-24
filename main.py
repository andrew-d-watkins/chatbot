import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tflearn
import tensorflow as tf
import random
import json
import nltk
import pickle
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

def get_data():
    #load training data
    with open('data.pickle', 'rb') as f:
        words, labels, training, output = pickle.load(f)

    return (words, labels, training, output)

def create_data():
    with open("intents.json") as file:
        intents = json.load(file)

    #create training data
    words = []
    labels = []
    docs_pattern = []
    docs_intent = []
    ignore_words = ['?']

    for intent in intents['intents']:
        for pattern in intent['patterns']:
            tokens = nltk.word_tokenize(pattern)
            words.extend(tokens)
            docs_pattern.append(tokens)
            docs_intent.append(intent["tag"])

            if intent['tag'] not in labels:
                labels.append(intent['tag'])

    words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]

    words = sorted(list(set(words)))
    labels = sorted(list(set(labels)))

    training = []
    output = []
    out_empty = [0] * len(labels)

    for X, doc in enumerate(docs_pattern):
        bag = []

        wrds = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
        output_row = out_empty[:]
        output_row[labels.index(docs_intent[X])] = 1

        training.append(bag)
        output.append(output_row)

    training = np.array(training)
    output = np.array(output)

    with open('data.pickle', 'wb') as f:
        pickle.dump((words, labels, training, output), f)

    return (words, labels, training, output)

def get_model(training, output):
    tf.reset_default_graph()

    net = tflearn.input_data(shape=[None, len(training[0])])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, len(output[0]), activation='softmax')
    net = tflearn.regression(net)

    try:
        model = tflearn.DNN(net)
        model.load('model.tflearn')
    except:
        model = tflearn.DNN(net)
        model.fit(training, output, n_epoch=1000, batch_size=8)
        model.save("model.tflearn")

    return model


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]

    return sentence_words

def bag_of_words(sentence, words):
    bag = [0] * len(words)

    sentence_clean = clean_up_sentence(sentence)

    for word in sentence_clean:
        for i, w in enumerate(words):
            if w == word:
                bag[i] = 1

    return np.array(bag)


def classify_input(input, words, labels, training, output):
    model = get_model(training, output)

    results = model.predict([bag_of_words(input, words)])
    tag_index = np.argmax(results)
    input_tag = labels[tag_index]

    return input_tag


def respond(input, words, labels, training, output):
    input_tag = classify_input(input, words, labels, training, output)

    with open("intents.json") as file:
        intents = json.load(file)

    for label in intents['intents']:
        if label['tag'] == input_tag:
            responses = label['responses']

    print(random.choice(responses))

def listen():
    try:
        words, labels, training, output = get_data()
    except:
        words, labels, training, output = create_data()

    while True:
        inp = input('You: ')
        if inp.lower() == 'quit':
            break

        respond(inp, words, labels, training, output)

def chat():
    print("Lets chat! (type quit to end conversation)")
    listen()

#starts the conversation
chat()
