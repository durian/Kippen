# -------------------------------------------------------------
# Train webcam images to detect chickens/no-chickens
# -------------------------------------------------------------
#
import sys
import os
import tensorflow as tf
import PIL
import matplotlib.pyplot as plt
import pandas as pd
import kippen4config as config
import argparse
import datetime
import numpy as np

# Train on empty / present from KIPPEN_TRAIN
# use feh  to sort
#   feh --action1 "echo %f;cp %f KIPPEN_TRAIN/NONE/" --action2 "echo %f;cp %f KIPPEN_TRAIN/PRESENT/"
#       -dF /a/webcams/kipcam1/out_img20200310*
#
# CUDA_VISIBLE_DEVICES="" python kippen4.py
#
# Checkpoints/saving
#   https://lambdalabs.com/blog/tensorflow-2-0-tutorial-03-saving-checkpoints/
#
# Small objects...
#  https://medium.com/datadriveninvestor/small-objects-detection-problem-c5b430996162
#  https://medium.com/@jonathan_hui/understanding-feature-pyramid-networks-for-object-detection-fpn-45b227b9106c
#  https://github.com/DetectionTeamUCAS/FPN_Tensorflow
#  https://www.paperswithcode.com/task/object-detection-in-aerial-images

parser = argparse.ArgumentParser()
parser.add_argument( '-b', "--batch_size",   type=int, default=28, help='Batch size' )
parser.add_argument( '-d', "--display_only", action='store_true', default=False, help='Only show plots' )
parser.add_argument( '-e', "--epochs",       type=int, default=10, help='Epochs' )
parser.add_argument( '-i', "--initial_epoch",type=int, default=0, help='Initial epoch' ) # TODO Implement!
parser.add_argument( '-p', "--show_plots",   action='store_true', default=False, help='Show plots' )
args = parser.parse_args()

# Augmentation is maybe not necessary... we have a static background
# on which we want to see chickens.
img_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                          ##rotation_range=20,
                                                          validation_split=0.4,
                                                          ##horizontal_flip=True,
                                                          ##width_shift_range=0.2,
                                                          #featurewise_center=True,
                                                          ##height_shift_range=0.2
)

train_generator = img_gen.flow_from_directory(
  config.train_dir,
  target_size=( config.image_height, config.image_width ),
  batch_size=args.batch_size,
  class_mode='categorical',
  shuffle=True,
  subset="training")
validation_generator = img_gen.flow_from_directory(
  config.train_dir,
  target_size=( config.image_height, config.image_width ),
  batch_size=args.batch_size,
  class_mode='categorical',
  subset="validation")


images, labels = next( train_generator )
print( "train_generator" )
print( "images.shape", images.shape )
print( "Images:", train_generator.samples )

images, labels = next(validation_generator)
print( "validation_generator" )
print( "images.shape", images.shape )
print( "Images:", validation_generator.samples )#.filenames )

cnt = 1 #train_generator.samples // 9
dim = 4
for a, b in train_generator:
  fig, axes = plt.subplots(nrows=dim, ncols=dim, figsize=(6,6))
  print( "a.shape", a.shape )
  print( "b.shape", b.shape )
  for i, p in enumerate(a[0:dim*dim]): # plot max dim^2
    ax  = axes[i//dim,i%dim] #plt.subplot(330 + 1 + i)
    cls = np.argmax(b[i])
    print( p.shape, b[i], cls )
    ax.imshow( p[:, :, :] ) # rgb 
    ax.set_xlabel( str(cls), c='r' )
    #plt.imshow( a[0, :, :, 0], cmap='gray' )
    #plt.imshow( a[1, :, :, 0], cmap='gray' )
  plt.tight_layout()
  plt.savefig( config.PREFIX+"train.png" )
  cnt -= 1
  if cnt <= 0:
    break
if args.display_only:
  plt.show(block=True)
  sys.exit(0)
if args.show_plots:
  plt.show(block=False)
  plt.pause(0.1) # actually shows them

# insipred by https://github.com/calmisential/TensorFlow2.0_Image_Classification/blob/master/models/vgg19.py
model = tf.keras.Sequential()
# 1
model.add(tf.keras.layers.Conv2D(filters=64,
                                 kernel_size=(3, 3),
                                 strides=1,
                                 padding='same',
                                 activation=tf.keras.activations.relu,
                                 input_shape=( config.image_height, config.image_width, config.channels )))
model.add(tf.keras.layers.Conv2D(filters=64,
                                 kernel_size=(3, 3),
                                 strides=1,
                                 padding='same',
                                 activation=tf.keras.activations.relu))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                    strides=2,
                                    padding='same'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.BatchNormalization())
# 2
model.add(tf.keras.layers.Conv2D(filters=128,
                                 kernel_size=(3, 3),
                                 strides=1,
                                 padding='same',
                                 activation=tf.keras.activations.relu))
model.add(tf.keras.layers.Conv2D(filters=128,
                                 kernel_size=(3, 3),
                                 strides=1,
                                 padding='same',
                                 activation=tf.keras.activations.relu))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                    strides=2,
                                    padding='same'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.BatchNormalization())
# 3
if True:
  model.add(tf.keras.layers.Conv2D(filters=256,
                                   kernel_size=(3, 3),
                                   strides=1,
                                   padding='same',
                                   activation=tf.keras.activations.relu))
  model.add(tf.keras.layers.Conv2D(filters=256,
                                   kernel_size=(3, 3),
                                   strides=1,
                                   padding='same',
                                   activation=tf.keras.activations.relu))
  model.add(tf.keras.layers.Conv2D(filters=256,
                                   kernel_size=(3, 3),
                                   strides=1,
                                   padding='same',
                                   activation=tf.keras.activations.relu))
  model.add(tf.keras.layers.Conv2D(filters=256,
                                   kernel_size=(3, 3),
                                   strides=1,
                                   padding='same',
                                   activation=tf.keras.activations.relu))
  model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                      strides=2,
                                      padding='same'))
  model.add(tf.keras.layers.Dropout(0.2))
  model.add(tf.keras.layers.BatchNormalization())
#
# 4
model.add(tf.keras.layers.Conv2D(filters=512,
                                 kernel_size=(3, 3),
                                 strides=1,
                                 padding='same',
                                 activation=tf.keras.activations.relu))
model.add(tf.keras.layers.Conv2D(filters=512,
                                 kernel_size=(3, 3),
                                 strides=1,
                                 padding='same',
                                 activation=tf.keras.activations.relu))
model.add(tf.keras.layers.Conv2D(filters=512,
                                 kernel_size=(3, 3),
                                 strides=1,
                                 padding='same',
                                 activation=tf.keras.activations.relu))
model.add(tf.keras.layers.Conv2D(filters=512,
                                 kernel_size=(3, 3),
                                 strides=1,
                                 padding='same',
                                 activation=tf.keras.activations.relu))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                    strides=2,
                                    padding='same'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.BatchNormalization())

# 5
model.add(tf.keras.layers.Conv2D(filters=512,
                                 kernel_size=(3, 3),
                                 strides=1,
                                 padding='same',
                                 activation=tf.keras.activations.relu))
model.add(tf.keras.layers.Conv2D(filters=512,
                                 kernel_size=(3, 3),
                                 strides=1,
                                 padding='same',
                                 activation=tf.keras.activations.relu))
model.add(tf.keras.layers.Conv2D(filters=512,
                                 kernel_size=(3, 3),
                                 strides=1,
                                 padding='same',
                                 activation=tf.keras.activations.relu))
model.add(tf.keras.layers.Conv2D(filters=512,
                                 kernel_size=(3, 3),
                                 strides=1,
                                 padding='same',
                                 activation=tf.keras.activations.relu))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                    strides=2,
                                    padding='same'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.BatchNormalization())

# Dense
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=4096, # number we get from lprevious layer
                                activation=tf.keras.activations.relu))
model.add(tf.keras.layers.Dense(units=2048,
                                activation=tf.keras.activations.relu))
model.add(tf.keras.layers.Dense(units=config.num_classes,
                                activation=tf.keras.activations.softmax))

# Custom accuracy class
class myAccuracy(tf.keras.metrics.Accuracy):
  def update_state(self, y_true, y_pred, sample_weight=None):
    y_true = tf.argmax(y_true,1)
    y_pred = tf.argmax(y_pred,1)
    return super(myAccuracy,self).update_state(y_true,y_pred,sample_weight)

class MyCustomCallback( tf.keras.callbacks.Callback ):
  #def on_train_batch_begin(self, batch, logs=None):
  #    print('Training: batch {} begins at {}'.format(batch, datetime.datetime.now().time()))
  
  #def on_train_batch_end(self, batch, logs=None):
  #    print('Training: batch {} ends at {}'.format(batch, datetime.datetime.now().time()))
  
  #def on_test_batch_begin(self, batch, logs=None):
  #    print('Evaluating: batch {} begins at {}'.format(batch, datetime.datetime.now().time()))
  
  #def on_test_batch_end(self, batch, logs=None):
  #    print('Evaluating: batch {} ends at {}'.format(batch, datetime.datetime.now().time()))
  
  #def on_predict_batch_end(self, batch, logs=None):
  #    print('Predicting: batch {} ends at {}'.format(batch, datetime.datetime.now().time()))
  
  def on_epoch_end(self, epoch, logs={}):
    #if args.show_plots:
    #  plt.pause(0.1) # small update, but puts in foreground, grmbl
    if epoch % 10 == 0:
      pass
      #print( '\n  epoch {:4d} Loss: {:.4f}'.format(epoch, logs["loss"]) )
      #model.save( config.save_dir )
      #vals = model.evaluate( validation_generator )
      #print( vals )
      #acc = vals[1] # loss, acc, mse (see compile(...)
      #print("The accuracy on validation set is: {:6.3f}%".format(acc*100))

  #def on_epoch_start(self, epoch, logs=None):
  #  if epoch % 10 == 0:
  #    print( '-> epoch {}.'.format(epoch) )

# Save all loss, accuracy, etc values to a CSV file
csv_logger = tf.keras.callbacks.CSVLogger( config.csv_log_file )

tensorboard = tf.keras.callbacks.TensorBoard(log_dir=config.tboard_dir,
                                             histogram_freq=0,
                                             update_freq="epoch",
                                             write_graph=False,
                                             write_images=False)

# TN FN etc
# https://stackoverflow.com/questions/47899463/how-to-extract-false-positive-false-negative-from-a-confusion-matrix-of-multicl

output_dir = config.chkpt_dir
if not os.path.exists(output_dir):
  os.makedirs(output_dir)
filepath = output_dir+"/model-{epoch:04d}-{accuracy:.4f}.hdf5"
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
  filepath,
  monitor='accuracy',
  verbose=0,
  save_best_only=True,
  save_weights_only=False,
  save_frequency=10
)

#sgd = tf.keras.optimizers.SGD(lr=0.001, momentum=0.9, nesterov=False)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=[
                'accuracy',
                #"mse",
                #myAccuracy(),
                #'TruePositives', 'TrueNegatives', #'FalsePositives', 'FalseNegatives',
                #"mae"
              ])
print( model.summary() )

history = model.fit( train_generator,
                     epochs=args.epochs,
                     verbose=1,
                     initial_epoch=args.initial_epoch,
                     validation_data=validation_generator,
                     callbacks=[
                       #MyCustomCallback(),
                       #checkpoint_callback,
                       csv_logger,
                       tensorboard,
                     ])
model.reset_metrics()
model.save( config.save_dir )
history_df = pd.DataFrame( history.history )
print( history_df )
if args.show_plots:
  history_df.plot(y=["loss", "accuracy", "val_loss", "val_accuracy"])
  plt.show()

