# -*- coding: utf-8 -*-
import cv2
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Conv2DTranspose,Add
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras



wh = 100
num = 300
trainData = np.zeros((num,wh,wh,3))
for i in range(num):
    try:
        path='C:\\Users\\ben20\\Desktop\\auto_coloring\\bald_eagle//egle ('+str(i+1)+').jpg'
        row = cv2.imread(path)
        row = cv2.cvtColor( row, cv2.COLOR_BGR2RGB )
        rowre = cv2.resize(row, (wh, wh),interpolation=cv2.INTER_AREA)
        trainData[i] = rowre.reshape(wh,wh,3)
    except:
        print('problem:',i)


def rgb2lab(img,num,w,h):
    '''
        img : 影像
        
        num : 影像數量
        
        w, h : 長, 寬
        
        功能 : 將影像從RGB轉成LAB，並將L與ＡＢ層分開。
    '''
    
    lab = np.zeros((num,w,h,3))
    l = np.zeros((num,w,h,1))
    ab = np.zeros((num,w,h,2))
    
    img = img.astype('float32')
    
    for i in range(num):

        
        lab[i] = cv2.cvtColor( img[i]/255.0, cv2.COLOR_RGB2LAB )  #RGB to LAB
    
    for j in range(num):
        l[j] = lab[j,:,:,0:1]
        ab[j] = lab[j,:,:,1:]
        
    return l,ab

def block(input_layer,k_num,k_size=3):
    '''
        input_layer : 輸入
        
        k_size : kernel 大小
        
        k_num : kernel 通道數
        
        功能 : 做多次convolution。
    '''
    
    C1 = Conv2D(k_num, k_size, activation='elu', padding = 'same')(input_layer)
    B = BatchNormalization()(C1)
    C2 = Conv2D(k_num, 1, activation='elu', padding = 'same')(B)
    B = BatchNormalization()(C2)
    C3 = Conv2D(k_num, k_size, activation='elu', padding = 'same')(B)
    out = Add()([C1,C3])
    
    return out
    
def encoder(input_img):
    '''
    input_img : 輸入的圖片

    功能 : 經過block後，將尺度縮小一半。
    '''
    
    En = BatchNormalization()(input_img)
    En = block(En,64)
    En = MaxPooling2D((2,2))(En)
    En2 = BatchNormalization()(En)
    En2 = block(En2,64)
    En3 = MaxPooling2D((2,2))(En2)
    
    return En3

def decoder(input_img):
    '''
    input_img : 輸入的圖片

    功能 : 經過block後，將尺度增加一半。
    '''
    
    De = BatchNormalization()(input_img)
    De = block(De,64)
    De = UpSampling2D()(De)
    De2 = BatchNormalization()(De)
    De2 = block(De2,64)
    De3 = UpSampling2D()(De2)
    
    return De3
    
trainL, trainAB = rgb2lab(trainData, num, wh, wh)    

data_gen = ImageDataGenerator()
train_data_generator = data_gen.flow(trainL,trainAB,batch_size=5)

inputs = Input(shape=(wh,wh,1))
m = encoder(inputs)
m = decoder(m)
m = Conv2D(2, 5, padding = 'same')(m)
model = Model(inputs,m)
model.summary()

opt = tensorflow.keras.optimizers.Adam(learning_rate=0.001)

model.compile(loss=['mse'],optimizer=opt)
his = model.fit(train_data_generator,epochs=1000)

epochs = range(1, len(his.history['loss'])+1)  
plt.plot(epochs, his.history['loss'],'-')  
plt.title('Training Loss')
plt.show()

'''
a = cv2.imread('bald_eagle//egle (600).jpg')
a = cv2.cvtColor( a, cv2.COLOR_BGR2RGB )
a = cv2.resize(a,(wh,wh))
test = np.zeros((1,wh,wh,3))
test[0] = a

l,ab = rgb2lab(test,1,wh,wh)

testL = np.zeros((1,wh,wh,1))
testL[0] = l

x = model.predict(testL)

org = np.zeros((1,wh,wh,3))

org[:,:,:,:1] = l
org[:,:,:,1:] = x

org = org.astype('float32')
t = cv2.cvtColor( org[0], cv2.COLOR_LAB2RGB )

plt.figure(figsize=(10, 5))
ax = plt.subplot(1, 2, 1)
plt.imshow(t.reshape(wh,wh,3))
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)


ax = plt.subplot(1, 2, 2)
plt.imshow(a.reshape(wh,wh,3))
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
'''