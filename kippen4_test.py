import tensorflow as tf
import kippen4config as config
import numpy as np
from sklearn.metrics import confusion_matrix
import sys

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255.0
)
test_generator = test_datagen.flow_from_directory("KIPPEN_TEST",
                                                  target_size=(config.image_height, config.image_width),
                                                  batch_size=1, # how to get all in one batch?
                                                  shuffle=False,
                                                  class_mode="categorical"
)

# Load the model
new_model = tf.keras.models.load_model( config.save_dir )
#new_model = tf.keras.models.load_model( "chkpt_kippen4/model-0830-1.0000.hdf5" )

# Get the accuracy on the test set
vals = new_model.evaluate( test_generator ) # three because in model.compile(...)
print( vals )
acc = vals[1] # loss, acc, categorial_accuracy, mse
print("The accuracy on test set is: {:6.3f}%".format(acc*100))

predictions = new_model.predict( test_generator ) # three because in model.compile(...)
#print( predictions )
filenames = test_generator.filenames
all_labels = []
all_predictions = []
for i in range(test_generator.samples):
    images, labels = next( test_generator )
    if np.argmax(labels[0]) == np.argmax(predictions[i]):
        equal_str = ""
    else:
        equal_str = " ERROR"
    print( filenames[i], labels[0], np.argmax(labels[0]), np.argmax(predictions[i]), predictions[i], equal_str )
    all_labels.append( np.argmax(labels[0]) )
    all_predictions.append( np.argmax(predictions[i]) )

#https://stackoverflow.com/questions/47899463/how-to-extract-false-positive-false-negative-from-a-confusion-matrix-of-multicl
print( all_predictions )
print()
print( all_labels )
print()
cm1 = confusion_matrix( all_predictions, all_labels )
print('Confusion Matrix : \n', cm1)
TP = np.diag(cm1)
FP = []
for i in range(config.num_classes):
    FP.append(sum(cm1[:,i]) - cm1[i,i])
FN = []
for i in range(config.num_classes):
    FN.append(sum(cm1[i,:]) - cm1[i,i])
TN = []
for i in range(config.num_classes):
    temp = np.delete(cm1, i, 0)   # delete ith row
    temp = np.delete(temp, i, 1)  # delete ith column
    TN.append(sum(sum(temp)))
print( "TP:{}, TN:{}, FP:{}, FN:{}".format(TP, TN, FP, FN) )
       
