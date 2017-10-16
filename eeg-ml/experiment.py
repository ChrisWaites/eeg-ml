import os
import sys
import open_bci_v3 as bci
import time
import collections
import numpy as np
import csv
import pynput
import pickle
import logging
import classification
import threading

# Constants
START_TIME = time.strftime('%m-%d-%Y_%H-%M-%S')
COM_PORT = 'COM3'
SAMPLE_RATE = bci.SAMPLE_RATE
EXPERIMENT_TIME = 5 * 60 # in seconds
WINDOW_LENGTH = 250
PREDICTION_WINDOW_LENGTH = 5
CHANGE_THRESHOLD = 0.93
TESTING_PROP = 0.2 # proportion of data to test classifiers over
FFT_FREQ = np.fft.rfftfreq(WINDOW_LENGTH, d=1.0/SAMPLE_RATE)
PLOT_FREQ = 3

# Globals
board = bci.OpenBCIBoard(port=COM_PORT)
eyes_closed = 0
window = collections.deque(maxlen=WINDOW_LENGTH)
prediction_window = collections.deque(maxlen=PREDICTION_WINDOW_LENGTH)
readings = []
feature_vectors = []
labels = []
current_state = 0
plotting = False

def calc_feature_vector():
	fft = np.fft.rfft(window)
	feature_vector = []
	for i in range(len(fft)):
		if FFT_FREQ[i] >= 8 and FFT_FREQ[i] <= 13:
			feature_vector.append(abs(fft[i]) ** 0.5)
	return feature_vector

def label_samples(sample):
	channel_reading = sample.channel_data[1]
	window.append(channel_reading)
	if len(window) == WINDOW_LENGTH:
		labels.append(eyes_closed)
        readings.append(channel_reading)
        feature_vector = calc_feature_vector()
		feature_vectors.append(feature_vector)

def predict_labels(sample, optimal_classifier):
	global current_state
	channel_reading = sample.channel_data[1]
	window.append(channel_reading)
	if len(window) == WINDOW_LENGTH:
		feature_vector = calc_feature_vector()
		result = (int(round(optimal_classifier.predict([feature_vector]))))
		prediction_window.append(result)
                if len(prediction_window) == PREDICTION_WINDOW_LENGTH:
                    #determine if the state should be changed
                    new_state = 1 - current_state
                    count = prediction_window.count(new_state)
                    if (count / len(prediction_window)) >= CHANGE_THRESHOLD:
                        #change the state
                        current_state = new_state
                        print(current_state)

def handle_key(key):
	global eyes_closed
	if key == pynput.keyboard.Key.space:
		eyes_closed = 0 if eyes_closed else 1
		print('Eye state change to ' + str(eyes_closed))

raw_input('Press enter to begin labelling.')

print("Starting key listener...")
with pynput.keyboard.Listener(on_press=handle_key) as key_listener:
    print("Done.\n")
    print("Starting labelling stream...")
    print("Current eye state 0. Press space to change state.")
    board.start_streaming(label_samples, lapse=EXPERIMENT_TIME)
    print("Done.\n")

if not os.path.exists(START_TIME):
    os.makedirs(START_TIME)

print("Writing collected data to file...")
with open(START_TIME + '/' + START_TIME + '_labelled.csv', 'wb') as file:
	file_writer = csv.writer(file)
	for i in range(len(feature_vectors)):
		file_writer.writerow([labels[i]] + feature_vectors[i])
with open(START_TIME + '/' + START_TIME + '_raw.csv', 'wb') as file:
    file_writer = csv.writer(file)
    for reading in readings:
        file_writer.writerow(reading)
print("Done.\n")

print("Starting classifier analysis...")
results = classification.perform_analysis(feature_vectors, labels, TESTING_PROP, START_TIME)
print("Done.\n")

optimal_classifier = max(results, key=lambda x: x[2])[1]

raw_input ('Press enter key to begin testing.')

print("Starting testing stream...")
board.start_streaming(lambda s: predict_labels(s, optimal_classifier), lapse=EXPERIMENT_TIME)
print("Done.\n")

