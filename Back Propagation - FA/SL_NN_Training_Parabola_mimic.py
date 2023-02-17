import os 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 
path   = os.getcwd()
Path_D = path + "\Data_Parabola" 
#####################################################
def activate(X_features, Y, file_name, type='reg', dir = os.getcwd()):
    """
    X_features      : (N*n) where N is number of data and n is number of feature
    Y               : output value (float for regression & label for classification)
    file_name       : reuired name for TFrecord file in string format
    batch_size      : batch size training for back propagation
    shuffle_buffer  : buffer for random feature vector selection
    dir             : the place we need to save (default as current directory)
    """
    writer = tf.io.TFRecordWriter('%s\%s.tfrecord' % (dir, file_name)) 
    for it in range(len(X_features)):
        feature = {}
        for i in range(len(X_features[0])):
            feature["f"+str(i)] = tf.train.Feature(float_list = tf.train.FloatList(value = [X_features[it][i]]))
        if type == 'reg':
            feature["y"] = tf.train.Feature(float_list = tf.train.FloatList(value = [Y[it]]))
        else:
            feature["y"] = tf.train.Feature(int64_list=tf.train.Int64List(value=[Y[it]]))
        tf_features   = tf.train.Features(feature = feature) 
        tf_example    = tf.train.Example(features = tf_features) 
        tf_serialized = tf_example.SerializeToString() 
        writer.write(tf_serialized)  # write serialized data 
    writer.close() 
###################################################
############### Data Creation #####################
###################################################
X = np.arange(-100,100,0.01)
X0 = X


X_train = np.array([X0.tolist()]).T
Z_train = X0**2
ss      = np.random.randint(0,len(X_train),int(len(X_train)*0.2))
X_val   = np.array(X_train[ss].tolist())
Z_val   = np.array(Z_train[ss].tolist())
ss      = np.random.randint(0,len(X_train),int(len(X_train)*0.1))
X_test  = np.array(X_train[ss].tolist())
Z_test  = np.array(Z_train[ss].tolist())

activate(X_train,Z_train,'Train_Data_parabola') 
activate(X_val, Z_val, 'Val_Data_parabola') 
activate(X_test,Z_test,'Test_Data_parabola') 

feature_map = { 'f0': tf.io.FixedLenFeature([1], dtype=tf.float32, default_value=[0.0]),
                'y': tf.io.FixedLenFeature([1], dtype=tf.float32, default_value=[0.0])} 

def _parse_function(serialized):
    # Parse the input tf.Example proto using the dictionary above.
    features   = tf.io.parse_single_example(serialized, feature_map)
    f0  = tf.cast(features['f0'], dtype=tf.float32)
    label  = tf.cast(features['y'], dtype=tf.float32)
    train_data = tf.concat([f0], axis=0)
    return train_data, label 


batch_size     = 128
Shuffel_buffer = 4096
TD_name        = "Train_Data_parabola.tfrecord"
VD_name        = "Val_Data_parabola.tfrecord"
TstD_name      = "Test_Data_parabola.tfrecord"
# max_epochs     = 10 
log_dir        = os.getcwd()
Train_Data = tf.data.TFRecordDataset(TD_name).map(_parse_function).batch(batch_size, drop_remainder=True).shuffle(Shuffel_buffer, reshuffle_each_iteration=True)#.prefetch(1)
Val_Data   = tf.data.TFRecordDataset(VD_name).map(_parse_function).batch(batch_size, drop_remainder=True).shuffle(Shuffel_buffer, reshuffle_each_iteration=True)
Test_Data  = tf.data.TFRecordDataset(TstD_name).map(_parse_function).batch(batch_size, drop_remainder=True).shuffle(Shuffel_buffer, reshuffle_each_iteration=True)
print(Val_Data)

###############################################
########## Model Architecture #################
###############################################
from tensorflow import keras 
from keras.layers import Dense 
from keras.losses import MeanSquaredError 
from keras.optimizers import Adam
import datetime 


model = tf.keras.Sequential([
    Dense(128, activation='relu', input_shape=(1,)),  # sigmoid , arctan_activation
    Dense(128, activation='relu'),
    Dense(1)])  # arctan_activation
model.summary()

### Configure a model for categorical classification.
model.compile(optimizer = Adam(learning_rate=0.05, beta_1=0.9, beta_2=0.999, epsilon=1e-07),
                loss=MeanSquaredError(),  
                metrics=['mse']) 
###############################################
############# Model Training ##################
###############################################
# date_time     = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") 
# check_path    =  Path_D + "\\check_points\\" + date_time 
# checkpoint_path = Path_D +"\\checkpoints\\" + date_time +"\\model-{epoch:02d}.hdf5"
# if not os.path.exists(check_path):
#     os.makedirs(check_path, exist_ok=True)

# # ES     = tf.keras.callbacks.EarlyStopping(monitor='loss',min_delta= 0.000001, patience=200) #,min_delta= 0.000001
# MCP    = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=False,save_freq='epoch',period=10)
# TB     = tf.keras.callbacks.TensorBoard(log_dir=log_dir, profile_batch='128,128', histogram_freq=1)  


# History = model.fit(Train_Data, epochs= 1500,
#                     steps_per_epoch = None,
#                     validation_data = Val_Data,
#                     callbacks=[MCP,TB],
#                     validation_steps=None, verbose=2) 

# model.save(Path_D + "\Model_parabola.h5")
# model.save_weights(Path_D+ "\Model_parabola.h5") 

# plt.figure(figsize=(9,6))
# plt.plot(History.history['mse'],label="Train Error",color="teal")
# plt.plot(History.history['val_mse'],label="Val Error", color = "crimson")
# plt.ylabel('MSE')
# plt.xlabel('Epochs')
# plt.legend(loc = "best")
# plt.grid()
# plt.title("Mean Square Error") 
# plt.savefig("TensorFlow_first_parabola.jpg",dpi=420)
# plt.show() 
######################################################
################ Model Evaluation ####################
######################################################

model.load_weights(Path_D+'\Model_parabola.h5') 


feature_map = { 'f0': tf.io.FixedLenFeature([1], dtype=tf.float32, default_value=[0.0]),
                'y': tf.io.FixedLenFeature([1], dtype=tf.float32, default_value=[0.0])} 

def _parse_function(serialized):
    # Parse the input tf.Example proto using the dictionary above.
    features   = tf.io.parse_single_example(serialized, feature_map)
    f0  = tf.cast(features['f0'], dtype=tf.float32)
    label  = tf.cast(features['y'], dtype=tf.float32)
    train_data = tf.concat([f0], axis=0)
    return train_data, label 

batch_size     = 96
Shuffel_buffer = 1024
TstD_name      = "Test_Data_parabola.tfrecord"
Test_Data  = tf.data.TFRecordDataset(TstD_name).map(_parse_function).batch(batch_size, drop_remainder=True).shuffle(Shuffel_buffer, reshuffle_each_iteration=True)
test_loss, test_mse = model.evaluate(Test_Data)

#####################################################
################ Sample Testing #####################
#####################################################

# # some testing
# test_inp         = []
# actual_op        = []
# for elm, l in Test_Data:
#     test_inp.extend(elm[:, 0:1])
#     actual_op.append(l)

# test_input = tf.reshape(test_inp, [len(test_inp), 1]).numpy()
X = np.arange(-100,100,0.01)
Y_a = X**2
Y_p = model.predict(X).flatten()

plt.figure(figsize=(9,6))
plt.plot(X,Y_a,color="crimson",linestyle="--")
plt.plot(X,Y_p,color="teal")
plt.xlabel("Actual Value")
plt.ylabel("Predicted Value") 
plt.title("Result Comparison")
plt.grid() 
plt.show()
########################################################
########################################################
ss =  tf.constant([[1.1]])#tf.reshape(test_inp[0], [1, 3]).numpy()
y  = model.predict(ss) 
print("Sathichutada thamara unakku thanda :",y) 
########################################################
########################################################





