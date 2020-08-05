import tensorflow as tf
import pandas as pd
import cv2
import numpy as np

def proprecess(x,y):
    x = tf.cast(x,dtype=tf.float32) / 255.0
    y = tf.cast(y,dtype=tf.int32)
    return x,y

df = pd.read_csv(r'F:\fish\FishClassification_FlyAI\data\input\FishClassification\train.csv')
img_path_list = df['image_path'].values
img_label_list = df['label'].values

path = 'F:\\fish\\FishClassification_FlyAI\\data\\input\\FishClassification\\'
print(len(img_path_list))
print(len(img_label_list))
train_x = []
train_y = []
test_x = []
test_y = []

for i in range(len(img_path_list)):
    img_path = path + img_path_list[i]
    imgs = cv2.imread(img_path,-1)
    imgs = cv2.resize(imgs,(128,128))
    if i % 10 == 0:
        test_x.append(imgs)
        test_y.append(img_label_list[i])
    else:
        train_x.append(imgs)
        train_y.append(img_label_list[i])

print(len(train_x) + len(test_x))
print(len(train_y) + len(test_y))

train_x = np.array(train_x,dtype=np.float32)
train_y = np.array(train_y,dtype=np.int32)
test_x = np.array(test_x,dtype=np.float32)
test_y = np.array(test_y,dtype=np.int32)

train_db = tf.data.Dataset.from_tensor_slices((train_x,train_y))
train_db = train_db.map(proprecess).shuffle(1000).batch(1)
test_db = tf.data.Dataset.from_tensor_slices((test_x,test_y))
test_db = test_db.map(proprecess).batch(32)

class Mymodel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=32,kernel_size=[3,3],padding='same',activation=tf.nn.relu)
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=[2,2],strides=[2,2])
        self.conv2 = tf.keras.layers.Conv2D(filters=64,kernel_size=[3,3],padding='same',activation=tf.nn.relu)
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=[2,2],strides=[2,2])
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(64,activation=tf.nn.relu)
        self.fc2 = tf.keras.layers.Dense(21,activation=tf.nn.softmax)
    def call(self,inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

model = Mymodel()
model.build(input_shape=(None,128,128,3))
model.summary()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)


for epoch in range(10):
    train_loss = 0
    train_num = 0
    for x,y in train_db:
        x = tf.reshape(x,[-1,128,128,3])
        with tf.GradientTape() as tape:
            pred = model(x)
            #print('pred:',pred)
            #pred = tf.cast(pred,dtype=tf.int32)
            #print('y:',y)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y,y_pred=pred)
            loss = tf.reduce_mean(loss)
        grads = tape.gradient(loss,model.trainable_variables)
        optimizer.apply_gradients(zip(grads,model.trainable_variables))
        train_loss += loss
        train_num += x.shape[0]
    loss = float(train_loss / train_num)

    total_correct = 0
    total_num = 0
    for x,y in test_db:
        x = tf.reshape(x,[-1,128,128,3])
        pred = model(x)
        pred = tf.argmax(pred,axis=1)
        pred = tf.cast(pred,dtype=tf.int32)
        correct = tf.equal(pred,y)
        correct = tf.reduce_sum(tf.cast(correct,dtype=tf.int32))
        total_correct += correct
        total_num += x.shape[0]
    accuracy = float(total_correct / total_num)
    print(epoch,'train_loss:',train_loss,'accuracy:',accuracy)

print('............................prediction..........................')
for x,y in test_db:
    img = x
    label = y
    break
img = tf.reshape(img,[-1,128,128,3])
pred = model(img)
pred = tf.argmax(pred,axis=1)
pred = tf.cast(pred,dtype=tf.int32)
print('pred:',pred)
print('label:',label)
print('The pred is equal the label',tf.equal(pred,label))