import tensorflow as tf
import PIL
import matplotlib.pyplot as plt
import pandas as pd
import kippen4config as config
import argparse
import datetime

# Train on empty / present from KIPPEN_TRAIN
# use feh  to sort
#   feh --action1 "echo %f;cp %f KIPPEN_TRAIN/NONE/" --action2 "echo %f;cp %f KIPPEN_TRAIN/PRESENT/"
#       -dF /a/webcams/kipcam1/out_img20200310*

parser = argparse.ArgumentParser()
parser.add_argument( '-e', "--epochs", type=int, default=10, help='Epochs' )
args = parser.parse_args()

img_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                          rotation_range=20,
                                                          validation_split=0.4,
                                                          horizontal_flip=True,
                                                          width_shift_range=0.2,
                                                          #featurewise_center=True,
                                                          height_shift_range=0.2
)
#images, labels = next(img_gen.flow_from_directory( config.train_dir ))
#print( labels )
#print(images.dtype, images.shape)
#print(labels.dtype, labels.shape)

train_generator = img_gen.flow_from_directory(
  config.train_dir,
  target_size=( config.image_height, config.image_width ),
  #target_size=(128, 128),
  batch_size=48,
  #class_mode='binary',
  class_mode='categorical',
  subset="training")
#img_gen.fit( train_generator )

images, labels = next( train_generator )
print( "train_generator" )
print( "images.shape", images.shape )

cnt = 1
for a, b in train_generator:
  print( "a.shape", a.shape )
  print( "b.shape", b.shape )
  for i, p in enumerate(a[0:9]): # plot max 9
    plt.subplot(330 + 1 + i)
    print( p.shape, b[i] )
    plt.imshow( p[:, :, :] ) # rgb
    #plt.imshow( a[0, :, :, 0], cmap='gray' )
    #plt.imshow( a[1, :, :, 0], cmap='gray' )
  plt.show(block=False)
  cnt -= 1
  if cnt <= 0:
    break
plt.pause(0.1) # actually shows them

#input_batch = next( train_generator ) 
#print( input_batch[0].shape )
#print( input_batch[1] )

# https://github.com/calmisential/TensorFlow2.0_Image_Classification/blob/master/models/vgg19.py
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
# 3
if False:
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
#    
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=128,
                                activation=tf.keras.activations.relu))
model.add(tf.keras.layers.Dense(units=128,
                                activation=tf.keras.activations.relu))
model.add(tf.keras.layers.Dense(units=config.num_classes,
                                activation=tf.keras.activations.softmax))

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
    if epoch % 10 == 0:
      print( 'epoch {:4d} Loss: {:.4f}'.format(epoch, logs["loss"]) )

  def on_epoch_start(self, epoch, logs=None):
    if epoch % 10 == 0:
      print( '-> epoch {}.'.format(epoch) )

csv_logger = tf.keras.callbacks.CSVLogger('kippen4_training.log')

# TN FN etc
# https://stackoverflow.com/questions/47899463/how-to-extract-false-positive-false-negative-from-a-confusion-matrix-of-multicl

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=[
                'accuracy',
                "mse",
                #myAccuracy(),
                #'TruePositives', 'TrueNegatives', 'FalsePositives', 'FalseNegatives',
                #"mae"
              ])

print( model.summary() )

history = model.fit( train_generator,
                     epochs=args.epochs,
                     verbose=1,
                     callbacks=[MyCustomCallback(), csv_logger]
)
model.save( config.save_dir )
history_df = pd.DataFrame( history.history )
print( history_df )
history_df.to_csv( "history.csv" )

history_df.plot(y=["loss", "accuracy", "mse"])
plt.show()

