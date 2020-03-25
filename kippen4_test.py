import tensorflow as tf
import kippen4config as config
import numpy as np

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255.0
)
test_generator = test_datagen.flow_from_directory("KIPPEN_TEST",
                                                    target_size=(config.image_height, config.image_width),
                                                    batch_size=1,
                                                    shuffle=False,
                                                    class_mode="categorical"
)


# Load the model
#new_model = tf.keras.models.load_model( config.save_dir )
new_model = tf.keras.models.load_model( "chkpt_kippen4/model-0830-1.0000.hdf5" )

# Get the accuracy on the test set
vals = new_model.evaluate( test_generator ) # three because in model.compile(...)
print( vals )
acc = vals[1] # loss, acc, categorial_accuracy, mse
print("The accuracy on test set is: {:6.3f}%".format(acc*100))
