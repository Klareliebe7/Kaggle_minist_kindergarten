

#   modify path if necessary   #
input_file = 'train.csv'
################################              这部分read data 我觉得不好      ###########################################################
#
#pixel_length = 783
#num_classes = 10
#input_len = 42000
#def read_data(input_file):
#    
#    data_x = [[[[0 for i in range(28)] for j in range(28)] ]for k in range(input_len)]
#    data_y = []
#    j = 0
#    with open(input_file) as f:
#        for num,line in enumerate(f):
#            if num == 0:
#                continue
#            line = line.split(',')
#            label, pixel = line[0],line[1:]
#            for heng in pixel:
#                for k in range(28):
#                    for i in range(28):
#                        data_x[j][0 ][k][i]=pixel[k*28+i]
#            label = int(label)
#            data_y.append(label)
#            j=j+1
#
#    return data_y , data_x## >>>  a = [1,2,3] c = [4,5,6,7,8]  zip(a,c) = [(1, 4), (2, 5), (3, 6)]
##    return data_y , data_x
#train_y, train_x = read_data(input_file)
##num,pixels = read_data(input_file)
##train_y, train_x = zip(*(data_train))  ##Python中的//是向下取整 5//2=2   -5//2=-3   zip(*data) =[(1, 2, 3), (4, 5, 6)]  unzip
#input_file = 'test.csv'
#test_y, test_x = read_data(input_file)
##test_y, test_x = zip(*(data_test))




##################################                    this part of reading is far better            #####################################################
import os
path1=os.path.abspath('.')   #表示当前所处的文件夹的绝对路径
print(path1)
path2 = os.getcwd()  # Get the current working directory (cwd)
print(path2)
files = os.listdir(path2)  # Get all the files in that directory
print("Files in %r: %s" % (path2, files))

import numpy as np

def read_data_train(input_file,col,row):
    data_x = []                           ################# 强烈推荐预留空间为一元list   这样更快，添加元素直接使用append
    data_y = []
    max_line = 0
    with open(input_file) as f:
        for num,line in enumerate(f):
            if num == 0:
                continue
            max_line = max_line+1
            line = line.split(',')
            label = line[0] 
            for word in line[1:]:
                data_x.append(word)
            data_y.append(label)
        data_x, data_y = np.array(data_x), np.array(data_y)             ########################将 list 转为 array， 并且将一维arry变为需要的格式
        print('shape of data_x'+str(data_x.shape))
        print('shape of data_y'+str(data_y.shape))
        data_x = data_x.reshape(max_line,1,col,row)
        print('shape of trasformed datax'+str(data_x.shape))
    return data_x,data_y




def read_data_test(input_file,col,row):
    data_x = []                           ################# 强烈推荐预留空间为一元list   这样更快，添加元素直接使用append
    max_line = 0
    with open(input_file) as f:
        for num,line in enumerate(f):
            if num == 0:
                continue
            max_line = max_line+1
            line = line.split(',')
            for word in line:
                data_x.append(word)
        data_x = np.array(data_x)             ########################将 list 转为 array， 并且将一维arry变为需要的格式
        print('shape of test_data'+str(data_x.shape))
        data_x = data_x.reshape(max_line,1,col,row)
        print('shape of trasformed data'+str(data_x.shape))
    return data_x

input_file = 'train.csv'
col = 28
row = 28
data_x, data_y = read_data_train(input_file,col,row)
input_file = 'test.csv'
col = 28
row = 28
pred_x = read_data_test(input_file,col,row)


##################################################      数据分割        ######################################################################
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.3, random_state=42)
#############################################################################################################################################
from keras.models import Sequential
import keras.layers
from keras.layers import Dense, Dropout, Flatten,Conv2D, MaxPooling2D

num_classes = 10

train_y = keras.utils.to_categorical(train_y, num_classes)
test_y = keras.utils.to_categorical(test_y, num_classes)


input_shape = (1,28,28)
batch_size = 128
epochs = 10
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 data_format="channels_first",
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), 
                 data_format="channels_first",
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
model.summary()
model.fit(train_x, train_y,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(test_x, test_y))
score = model.evaluate(test_x, test_y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
############################################################   模型保存 #############################################################
from keras.models import load_model
model.save('my_model.h5')
del model

model = load_model('my_model.h5')
############################################################ 预测      #############################################################
preds = model.predict(pred_x)
print(preds)
prediction =[]
for pred in preds:
    prediction.append(np.argmax(pred))                            #####    np.argmax!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#############################################################    写文件   ###########################################################
f = open('D:\\Kejian\\kaggle\\MINIST\\sub.csv', 'w')
f.write('ImageId,Label\n')
for arg,pred in enumerate(prediction):
    a = str(arg+1)+','+str(pred)+'\n'
    f.write(a)
f.close()
#data_x = [[[[0 for i in range(28)] for j in range(28)] ]for k in range(input_len)]# THIS IS A LIST!!!!!!!!!!! SO IT HAS LENGHT BUT NO SHAPE
#array_x=np.array(data_x)                                                          # THIS IS AN ARRAY!!!!!!!!!! IT CAN HAVE SHAPE
#print(np.shape(array_x))