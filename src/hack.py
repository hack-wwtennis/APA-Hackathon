import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import math

# (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

full_data = pd.read_excel('output2.xlsx', sheet_name='Sheet1')

full_data_train = full_data.loc[:500]
full_data_test = full_data.loc[500:]

rows, cols = full_data.shape

# def normalize(data):

def getfloat(str):
	if str == 'nan':
		return nan
	try:
		return float(str)
	except ValueError:
		return 0.0



input_columns = ['Paw Temperature', 'Gum Color', 'Attitude', 'Co-morbidities noted TODAY']
inputs = [[] for r in range(rows)]
outputs = []
normalize=[0,0,0,0,0,1,0,1,0,1,0,0,1,0,0,0,0,0,1,0,1,0,1,0,1,0,0,1,0,1,0,1,0,1,0,1,0,1,0,0,0,1,0,0,1,0,1,0,1,0,0]
# print(len(normalize))
# exit()
for r in range(rows):
	outputs.append(full_data.loc[r]['outcome'])
	inputs[r].append(full_data.loc[r]['Paw Temperature'])
	inputs[r].append(full_data.loc[r]['Gum Color'] / 3.0)
	inputs[r].append((full_data.loc[r]['Attitude'] - 1) / 3.0)
	inputs[r].append(0 if str(full_data.loc[r]['Co-morbidities noted TODAY']) == 'nan' else 1.0)
	inputs[r].append(full_data.loc[r]['On Distemper Watch? (only mark on shift watch started)'])
	tmp = full_data.loc[r]['Baytril mL (Strength:100mg/mL) SQ ONLY']
	inputs[r].append(0 if math.isnan(tmp) else tmp) #normalize
	inputs[r].append(0 if math.isnan(tmp) else 1)
	tmp = full_data.loc[r]['Polyflex mL (Strength: 200mg/mL) SQ ONLY']
	inputs[r].append(0 if math.isnan(tmp) else tmp) #normalize
	inputs[r].append(0 if math.isnan(tmp) else 1)
	tmp = full_data.loc[r]['Metoclopromide mL (Strength: 5mg/mL)']
	inputs[r].append(0 if math.isnan(tmp) else tmp) #normalize
	inputs[r].append(0 if math.isnan(tmp) else 1)
	tmp = str(full_data.loc[r]['Metoclopromide Route'])
	inputs[r].append(0 if tmp=='nan' else (.5 if tmp=='SQ' else 1.0))
	tmp = full_data.loc[r]['Cerenia mL ']
	inputs[r].append(0 if math.isnan(tmp) else tmp) #normalize
	inputs[r].append(0 if math.isnan(tmp) else 1)
	tmp = str(full_data.loc[r]['Cerenia Route'])
	inputs[r].append(0 if tmp=='nan' else (.5 if tmp=='SQ' else 1.0))
	inputs[r].append(full_data.loc[r]['Vomiting'] / 3.0)
	inputs[r].append(full_data.loc[r]['Appetite'] / 3.0)
	inputs[r].append(full_data.loc[r]['Feces'] / 4.0)
	tmp = full_data.loc[r]['Packed Cell Volume % (not required each shift)']
	inputs[r].append(0 if math.isnan(tmp) else tmp) #normalize
	inputs[r].append(0 if math.isnan(tmp) else 1)
	tmp = getfloat(full_data.loc[r]['Blood Glucose (not required each shift)'])
	inputs[r].append(0 if math.isnan(tmp) else tmp) #normalize
	inputs[r].append(0 if math.isnan(tmp) else 1)
	tmp = full_data.loc[r]['Total Protein (not required each shift)']
	inputs[r].append(0 if math.isnan(tmp) else tmp) #normalize
	inputs[r].append(0 if math.isnan(tmp) else 1)
	tmp = full_data.loc[r]['Temperature F (not required each shift)']
	inputs[r].append(0 if math.isnan(tmp) else tmp) #normalize
	inputs[r].append(0 if math.isnan(tmp) else 1)
	inputs[r].append(full_data.loc[r]['Drinking Water'])
	tmp = full_data.loc[r]['Cefazolin mL (1 gram/vial) IV ONLY']
	inputs[r].append(0 if math.isnan(tmp) else tmp) #normalize
	inputs[r].append(0 if math.isnan(tmp) else 1)
	tmp = full_data.loc[r]['Hetastarch Rate mL/hr (6% Concentrate) IV ONLY']
	inputs[r].append(0 if math.isnan(tmp) else tmp) #normalize
	inputs[r].append(0 if math.isnan(tmp) else 1)
	tmp = full_data.loc[r]['Oral Dextrose mL (50% concentrate)']
	inputs[r].append(0 if math.isnan(tmp) else tmp) #normalize
	inputs[r].append(0 if math.isnan(tmp) else 1)
	tmp = full_data.loc[r]['Panacur mL (100mg/mL) ORAL ONLY']
	inputs[r].append(0 if math.isnan(tmp) else tmp) #normalize
	inputs[r].append(0 if math.isnan(tmp) else 1)
	tmp = full_data.loc[r]['Marquis Paste ORAL ONLY']
	inputs[r].append(0 if math.isnan(tmp) else tmp) #normalize
	inputs[r].append(0 if math.isnan(tmp) else 1)
	tmp = full_data.loc[r]['Amount of SQ Fluids Administered mL']
	inputs[r].append(0 if math.isnan(tmp) else tmp) #normalize
	inputs[r].append(0 if math.isnan(tmp) else 1)
	tmp = full_data.loc[r]['Anzemet mL (20mg/mL) IV ONLY']
	inputs[r].append(0 if math.isnan(tmp) else tmp) #normalize
	inputs[r].append(0 if math.isnan(tmp) else 1)
	tmp = full_data.loc[r]['Famotidine mL (10mg/mL) ']
	inputs[r].append(0 if math.isnan(tmp) else tmp) #normalize
	inputs[r].append(0 if math.isnan(tmp) else 1)
	tmp = str(full_data.loc[r]['Famotidine Route'])
	inputs[r].append(0 if tmp=='nan' else (.5 if tmp=='SQ' else 1.0))
	tmp = full_data.loc[r]['Amount of IV Fluids Administered mL/hr']
	inputs[r].append(0 if math.isnan(tmp) else tmp) #normalize
	inputs[r].append(0 if math.isnan(tmp) else 1)
	tmp = full_data.loc[r]['Oral Nutrical mL']
	inputs[r].append(0 if math.isnan(tmp) else tmp) #normalize
	inputs[r].append(0 if math.isnan(tmp) else 1)
	tmp = full_data.loc[r]['Hetastarch Dosage mL (6% Concentrate) IV ONLY']
	inputs[r].append(0 if math.isnan(tmp) else tmp) #normalize
	inputs[r].append(0 if math.isnan(tmp) else 1)
	tmp = str(full_data.loc[r]['Dominant Route'])
	inputs[r].append(0 if tmp=='nan' else (.5 if tmp=='SQ' else 1.0))

train_inputs = np.array(inputs[:500]);
test_inputs = np.array(inputs[500:]);
train_outputs = np.array(outputs[:500]);
test_outputs = np.array(outputs[500:]);

trans = [[train_inputs[j][i] for j in range(len(train_inputs))] for i in range(len(train_inputs[0]))]

means = [np.mean(row) for row in trans]
stds = [np.std(row) for row in trans]

# print([train_inputs[r][39] for r in range(len(train_inputs))])

for r in range(len(train_inputs)):
	for c in range(len(train_inputs[r])):
		if normalize[c] == 1:
			train_inputs[r][c] = (train_inputs[r][c] - means[c]) / stds[c]

for r in range(len(test_inputs)):
	for c in range(len(test_inputs[r])):
		if normalize[c] == 1:
			test_inputs[r][c] = (test_inputs[r][c] - means[c]) / stds[c]

# for r in range(len(train_inputs)):
# 	for c in range(len(train_inputs[r])):
# 		if math.isnan(train_inputs[r][c]):
# 			print(r, c)

# print(train_inputs)
# print(test_inputs)



def create_model():
	model = tf.keras.models.Sequential([
		keras.layers.Dense(32, kernel_regularizer=keras.regularizers.l2(0.001), activation=tf.nn.relu, input_shape=(51,)),
		keras.layers.Dense(64, kernel_regularizer=keras.regularizers.l2(0.001), activation=tf.nn.sigmoid),
		keras.layers.Dense(32, kernel_regularizer=keras.regularizers.l2(0.001), activation=tf.nn.relu),
		keras.layers.Dense(1, activation=tf.nn.sigmoid)
	])


	model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

	return model

model = create_model()
model.summary()
model.fit(train_inputs, train_outputs, epochs=30)


results = model.evaluate(test_inputs, test_outputs)

print(results)