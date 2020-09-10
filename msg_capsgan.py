

#Image lib
import cv2
# math libraries
import numpy as np
import scipy.misc
# ml libraries
import tensorflow as tf
from keras import layers, models, optimizers
from keras import backend as K
from keras.utils import to_categorical
from keras.datasets import mnist, cifar10
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Lambda, concatenate, Multiply
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Add
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras import callbacks
import datetime
# visualization
import skimage
from skimage import data, color, exposure
from skimage.transform import resize
import matplotlib.pyplot as plt
import math
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
from keras.applications import VGG19


# sys and helpers
import sys
import os
import glob
from tqdm import tqdm

print('Modules imported.')

class DataLoader3_all():
    def __init__(self, dataset_name, img_res=(128, 128)):
        self.dataset_name = dataset_name
        self.img_res = img_res

    def load_data(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "test"
        
        path = glob.glob('/content/drive/My Drive/CelebA_sample/CelebA/*')

        batch_images = np.random.choice(path, size=batch_size)

        imgs_hr = []
        imgs_lr = []
        imgs_32 = []
        imgs_64 = []
        for img_path in batch_images:
            imgl = self.imread(img_path)
            img = imgl[20:218-20,:]
            h, w = self.img_res
            low_h, low_w = int(h / 8), int(w / 8)

#            img_hr = scipy.misc.imresize(img, self.img_res)
 #           img_lr = scipy.misc.imresize(img, (low_h, low_w))
  #          img_32 = scipy.misc.imresize(img, (32, 32))
   #         img_64 = scipy.misc.imresize(img, (64, 64))

            img_hr = cv2.resize(img, self.img_res)
            img_lr = cv2.resize(img, (low_h, low_w))
            img_32 = cv2.resize(img, (32, 32))
            img_64 = cv2.resize(img, (64, 64))
            # If training => do random flip
            if not is_testing and np.random.random() < 0.5:
                img_hr = np.fliplr(img_hr)
                img_lr = np.fliplr(img_lr)
                img_32 = np.fliplr(img_32)
                img_64 = np.fliplr(img_64)

            imgs_hr.append(img_hr)
            imgs_lr.append(img_lr)
            imgs_32.append(img_32)
            imgs_64.append(img_64)

        imgs_hr = np.array(imgs_hr) / 127.5 - 1.
        imgs_lr = np.array(imgs_lr) / 127.5 - 1.
        imgs_32 = np.array(imgs_32) / 127.5 - 1.
        imgs_64 = np.array(imgs_64) / 127.5 - 1.

        return imgs_hr, imgs_lr , imgs_32, imgs_64


    def imread(self, path):
        return plt.imread(path).astype(np.float)
    
    def load_test(self,path):
        img = self.imread(path)
        img_hr = scipy.misc.imresize(img, (256,256))  

# Configure data loader
dataset_name = 'img_align_celeba'
hr_height = 128
hr_width = 128

data_loader = DataLoader3_all(dataset_name=dataset_name,
                              img_res=(hr_height, hr_width))

def all_psnr(imageA, imageB):
    psnrs = []
    for ii in range(len(imageA)):
        psnrs.append(psnr(imageA[ii],imageB[ii]))
    return np.mean(psnrs)

def all_ssim(imageA, imageB):
    psnrs = []
    for ii in range(len(imageA)):
        psnrs.append(ssim(imageA[ii],imageB[ii],multichannel =True))
    return np.mean(psnrs)

        
# squash function of capsule layers, borrowed from Xifeng Guo's implementation of Keras CapsNet `https://github.com/XifengGuo/CapsNet-Keras`
def squash(vectors, axis=-1):
    """
    The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
    :param vectors: some vectors to be squashed, N-dim tensor
    :param axis: the axis to squash
    :return: a Tensor with same shape as input vectors
    """
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
    return scale * vectors

# device check
from tensorflow.python.client import device_lib
print('Devices:', device_lib.list_local_devices())

# GPU check
if not tf.test.gpu_device_name():
    print('No GPU found.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))



def build_vgg(hr_shape):
    """
    Builds a pre-trained VGG19 model that outputs image features extracted at the
    third block of the model
    """
    
    vgg = VGG19(weights="imagenet")
    # Set outputs to outputs of last conv. layer in block 3
    # See architecture at: https://github.com/keras-team/keras/blob/master/keras/applications/vgg19.py
    vgg.outputs = [vgg.layers[9].output]

    img = Input(hr_shape)

    # Extract image features
    img_features = vgg(img)

    return Model(img, img_features)

optimizer = Adam(0.0003, 0.5)
vgg = build_vgg((128,128,3))
vgg.trainable = False
vgg.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

vgg64 = build_vgg((64,64,3))
vgg64.trainable = False
vgg64.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

vgg32 = build_vgg((32,32,3))
vgg32.trainable = False
vgg32.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

# discriminator structure
def build_discriminator():

    # depending on dataset we define input shape for our network
    img = Input(shape=(128, 128, 3))
    img64 = Input(shape=(64, 64, 3))
    img32 = Input(shape=(32, 32, 3))
    
    x1 = Conv2D(filters=128, kernel_size=9, strides=2, padding='same', name='conv0')(img)
    img64f = Conv2D(filters=128, kernel_size=9, strides=1, padding='same', name='convimg0')(img64)
    x2 = concatenate([x1,img64f],axis=-1)
    x3 = Conv2D(filters=128, kernel_size=9, strides=2, padding='same', name='conv1')(x2)
    img32f = Conv2D(filters=128, kernel_size=9, strides=1, padding='same', name='convimg1')(img32)
    x4 = concatenate([x3,img32f],axis=-1)
    # first typical convlayer outputs a 20x20x256 matrix
    x5 = Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', name='conv2')(x4)
    x6 = LeakyReLU()(x5)
    
    # original 'Dynamic Routing Between Capsules' paper does not include the batch norm layer after the first conv group
    x = BatchNormalization(momentum=0.8)(x6)

    
    #
    # primarycaps coming first
    #
    
    # filters 256 (n_vectors=8 * channels=32)
    x = Conv2D(filters=8 * 32, kernel_size=9, strides=2, padding='valid', name='primarycap_conv2')(x)
    
    #Flatten to add dense layer
    x = Flatten()(x)
    
    # reshape into the 8D vector for all 32 feature maps combined
    # (primary capsule has collections of activations which denote orientation of the digit
    # while intensity of the vector which denotes the presence of the digit)
    x = Reshape(target_shape=[-1, 8], name='primarycap_reshape')(x)
    
    # the purpose is to output a number between 0 and 1 for each capsule where the length of the input decides the amount
    x = Lambda(squash, name='primarycap_squash')(x)
    x = BatchNormalization(momentum=0.8)(x)


    #
    # digitcaps are here
    #

    x = Flatten()(x)

    uhat = Dense(160, kernel_initializer='he_normal', bias_initializer='zeros', name='uhat_digitcaps')(x)
    
    # c = coupling coefficient (softmax over the bias weights, log prior) | "the coupling coefficients between capsule (i) and all the capsules in the layer above sum to 1"
    # we treat the coupling coefficiant as a softmax over bias weights from the previous dense layer
    c = Activation('softmax', name='softmax_digitcaps1')(uhat) # softmax will make sure that each weight c_ij is a non-negative number and their sum equals to one
    
    # s_j (output of the current capsule level) = uhat * c
    c = Dense(160)(c) # compute s_j
    x = Multiply()([uhat, c])
    """
    NOTE: Squashing the capsule outputs creates severe blurry artifacts, thus we replace it with Leaky ReLu.
    """
    s_j = LeakyReLU()(x)


    #
    # we will repeat the routing part 2 more times (num_routing=3) to unfold the loop
    #
    c = Activation('softmax', name='softmax_digitcaps2')(s_j) # softmax will make sure that each weight c_ij is a non-negative number and their sum equals to one
    c = Dense(160)(c) # compute s_j
    x = Multiply()([uhat, c])
    s_j = LeakyReLU()(x)

    c = Activation('softmax', name='softmax_digitcaps3')(s_j) # softmax will make sure that each weight c_ij is a non-negative number and their sum equals to one
    c = Dense(160)(c) # compute s_j
    x = Multiply()([uhat, c])
    s_j = LeakyReLU()(x)

    pred = Dense(25*25, activation='sigmoid')(s_j)
    patch = Reshape(target_shape=[-1, 25], name='last_out_reshape')(pred)
    
    
    return Model([img,img64,img32], patch)



# build and compile the discriminator
discriminator = build_discriminator()
print('DISCRIMINATOR:')
discriminator.summary()
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#discriminator.load_weights('images_MSG/wdis6400.h5')
discriminator.compile(loss='binary_crossentropy', optimizer=sgd)

# generator structure   
def build_generator():
    from keras.applications import VGG19
    vgg = VGG19(weights="imagenet")
    vgg.outputs = [vgg.layers[9].output]
    vgg.trainable =False
    
    def deconv2d(layer_input,num=256):
        """Layers used during upsampling"""
        u = UpSampling2D(size=2)(layer_input)
        u = Conv2D(num, kernel_size=3, strides=1, padding='same')(u)
        u = Activation('relu')(u)
        return u
    
    def residual_block(layer_input, filters):
        """Residual block described in paper"""
        d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(layer_input)
        d = Activation('relu')(d)
        d = BatchNormalization(momentum=0.8)(d)
        d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(d)
        d = BatchNormalization(momentum=0.8)(d)
        d = Add()([d, layer_input])
        return d

    img_lr_in = Input(shape=(16,16,3))
    img_bilinear =  Lambda(lambda image: tf.image.resize_images(image,(128, 128),method = tf.image.ResizeMethod.BICUBIC,align_corners = True, preserve_aspect_ratio = True))(img_lr_in) 
    img_bilinear_64 =  Lambda(lambda image: tf.image.resize_images(image,(64, 64),method = tf.image.ResizeMethod.BICUBIC,align_corners = True, preserve_aspect_ratio = True))(img_lr_in)
    img_bilinear_32 =  Lambda(lambda image: tf.image.resize_images(image,(32, 32),method = tf.image.ResizeMethod.BICUBIC,align_corners = True, preserve_aspect_ratio = True))(img_lr_in)
    vgg_features = vgg(img_bilinear)
    vgg_features_cnn = Conv2D(64, kernel_size=3, strides=1, padding='same',name = "vgg_features_cnn")(vgg_features)


    pd = Conv2D(64, kernel_size=3, strides=1, padding='same')(img_lr_in)
    x = residual_block(pd,64)
    x = deconv2d(x,num=64)
    xconcat = concatenate([x,vgg_features_cnn],axis=-1,name='concatenation')
    pd = Conv2D(64, kernel_size=3, strides=1, padding='same')(xconcat)
    x = residual_block(pd,64)
    x_out_32 = Conv2D(3, kernel_size=3, strides=1, padding='same')(x)
    x_out_32 = Activation('tanh')(x_out_32)
    x_32_add = Add()([x_out_32, img_bilinear_32])
    x_32 = Activation('tanh')(x_32_add)
    x = deconv2d(x)
    pd = Conv2D(64, kernel_size=3, strides=1, padding='same')(x)
    x = residual_block(pd,64)
    x_out_64 = Conv2D(3, kernel_size=3, strides=1, padding='same')(x)
    x_out_64 = Activation('tanh')(x_out_64)
    x_64_add = Add()([x_out_64, img_bilinear_64])
    x_64 = Activation('tanh')(x_64_add)
    x = deconv2d(x)
    x = Conv2D(3, kernel_size=3, strides=1, padding='same')(x)
    x_out_pre = Activation('tanh')(x)  
    x_out_add = Add()([x_out_pre, img_bilinear])
    x_out = Activation('tanh')(x_out_add) 
    
    return Model([img_lr_in],[x_out,x_64,x_32])
    
    
# build and compile the generator
generator = build_generator()
print('GENERATOR:')
generator.summary()
#generator.load_weights('images_MSG/wgen6400.h5')
generator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
model_json = generator.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model_json = generator.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# feeding noise to generator
z = Input(shape=(16,16,3))
imgf = generator(z)
fake_features = vgg(imgf[0])
fake_features_1 = vgg64(imgf[1])
fake_features_2 = vgg32(imgf[2])


# for the combined model we will only train the generator
discriminator.trainable = False

# try to discriminate generated images
valid = discriminator(imgf)

# the combined model (stacked generator and discriminator) takes
# noise as input => generates images => determines validity 
combined = Model(z, [valid,fake_features,fake_features_1,fake_features_2])
print('COMBINED:')
combined.summary()
combined.compile(loss=['binary_crossentropy', 'mse', 'mse', 'mse'], 
                 loss_weights=[3e-3, 0.001,0.001,1],
                 optimizer=Adam(0.0002, 0.5))

# loss values for further plotting
D_L_REAL = []
D_L_FAKE = []
D_L = []
D_ACC = []
G_L = []

def train(dataset_title, epochs, batch_size=32, save_interval=50):
        start_time = datetime.datetime.now()
        half_batch = int(batch_size / 2)

        for epoch in range(0,50000):
            '''
			#Progressive Weight Loss Adjustment
            if epoch == 600:
                combined.compile(loss=['binary_crossentropy', 'mse'], 
                 loss_weights=[1e-2, 1],
                 optimizer=Adam(0.0002, 0.5))    
            # ---------------------
            if epoch == 1500:
                combined.compile(loss=['binary_crossentropy', 'mse'], 
                 loss_weights=[1e-1, 1],
                 optimizer=Adam(0.0002, 0.5)) 
            if epoch == 2500:
                combined.compile(loss=['binary_crossentropy', 'mse'], 
                 loss_weights=[1e-2, 1],
                 optimizer=Adam(0.0002, 0.5))
            if epoch == 3500:
                combined.compile(loss=['binary_crossentropy', 'mse'], 
                 loss_weights=[1e-1, 1],
                 optimizer=Adam(0.0002, 0.5))  
            '''
            #  Train Discriminator
            # ---------------------

            # select a random half batch of images
            imgs_hr, imgs_lr, imgs_32, imgs_64 = data_loader.load_data(half_batch)


            # generate a half batch of new images
            gen_imgs = generator.predict(imgs_lr)

            # train the discriminator by feeding both real and fake (generated) images one by one
            d_loss_real = discriminator.train_on_batch([imgs_hr,imgs_64,imgs_32], np.random.uniform(low=0.8, high=1.2, size=(half_batch, 25,25))) # 0.9 for label smoothing
            d_loss_fake = discriminator.train_on_batch(gen_imgs, np.random.uniform(low=0.0, high=0.3, size=(half_batch, 25,25)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


            # ---------------------
            #  Train Generator
            # ---------------------


            imgs_hr, imgs_lr, imgs_32, imgs_64 = data_loader.load_data(batch_size)
            # the generator wants the discriminator to label the generated samples
            # as valid (ones)
            image_features = vgg.predict(imgs_hr)
            image_features_1 = vgg64.predict(imgs_64)
            image_features_2 = vgg32.predict(imgs_32)

            # train the generator
            g_loss = combined.train_on_batch(imgs_lr, [np.ones((batch_size,25,25)),image_features,image_features_1,image_features_2])
            elapsed_time = datetime.datetime.now() - start_time
            
            # Plot the progress
            print ("%d time: %s ,[D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch,elapsed_time, d_loss, 100*d_loss, g_loss[0]))
            D_L_REAL.append(d_loss_real)
            D_L_FAKE.append(d_loss_fake)
            D_L.append(d_loss)
            D_ACC.append(d_loss)
            G_L.append(g_loss)

            # if at save interval => save generated image samples
            if epoch % save_interval == 0:
                sample_images(epoch)
                generator.save_weights("images_MSG/wgen%d.h5" % (epoch))
                discriminator.save_weights("images_MSG/wdis%d.h5" % (epoch))

def sample_images(epoch):
    os.makedirs('images_MSG/%s' % dataset_name, exist_ok=True)
    r, c = 2, 5

    imgs_hr, imgs_lr,_,_ = data_loader.load_data(batch_size=2, is_testing=True)
    fake_hr = generator.predict(imgs_lr)
    imgs_hr_inp = imgs_hr
    imgs_lr_inp = imgs_lr
    # Rescale images 0 - 1
    imgs_lr = 0.5 * imgs_lr + 0.5
    fake_hr[0] = 0.5 * fake_hr[0] + 0.5
    fake_hr[1] = 0.5 * fake_hr[1] + 0.5
    fake_hr[2] = 0.5 * fake_hr[2] + 0.5
    imgs_hr = 0.5 * imgs_hr + 0.5

    # Save generated images and the high resolution originals
    titles = ['Original','128x128']
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for row in range(r):
        for col, image in enumerate([imgs_hr,fake_hr[0]]):
            axs[row, col+3].imshow(imgs_lr[row])
            axs[row, col+2].imshow(fake_hr[2][row])
            axs[row, col+1].imshow(fake_hr[1][row])
            axs[row, col].imshow(image[row])
            axs[row, col+1].set_title('64x64')
            axs[row, col+2].set_title('32x32')
            axs[row, col+3].set_title('input')
            axs[row, col].set_title(titles[col])
            axs[row, col].axis('off')
            axs[row, col+1].axis('off')
            axs[row, col+2].axis('off')
            axs[row, col+3].axis('off')
        cnt += 1
    txt = ("psnr = %f - ssim = %f" % (all_psnr(fake_hr[0],imgs_hr), all_ssim(fake_hr[0],imgs_hr)))
    fig.text(.5, .05, txt, ha='center')
    fig.savefig("images_MSG/%s/%d.png" % (dataset_name, epoch))
    plt.close()

    # Save low resolution images for comparison
    for i in range(r):
        fig = plt.figure()
        plt.imshow(imgs_lr[i])
        fig.savefig('images_MSG/%s/%d_lowres%d.png' % (dataset_name, epoch, i))
        plt.close()
    
    print(all_psnr(fake_hr[0],imgs_hr))
    print(all_ssim(fake_hr[0],imgs_hr))
    if epoch<5000 and epoch>1000:
        combined.compile(loss=['binary_crossentropy', 'mse', 'mse', 'mse'], 
                         loss_weights=[3e-3, 0.001,(epoch - 999)/4000,1],
                         optimizer=Adam(0.0002, 0.5))          
        
    if epoch<10000 and epoch>5000:
        combined.compile(loss=['binary_crossentropy', 'mse', 'mse', 'mse'], 
                         loss_weights=[3e-3, (epoch - 4999)/5000,1,1],
                         optimizer=Adam(0.0002, 0.5))
    if epoch<15000 and epoch>10000:
        combined.compile(loss=['binary_crossentropy', 'mse', 'mse', 'mse'], 
                         loss_weights=[3e-3 + 2e-3*(epoch - 9999)/5000, 1,1,1],
                         optimizer=Adam(0.0002, 0.5))

    return imgs_lr_inp,imgs_hr_inp

history = train('cifar10', epochs=30000, batch_size=32, save_interval=50)
#generator.save('mnist_model.h5')
#generator.save('cifar10_model.h5')


plt.plot(D_L)
plt.title('Discriminator results (MNIST)')
plt.xlabel('Epochs')
plt.ylabel('Discriminator Loss (blue), Discriminator Accuracy (orange)')
plt.legend(['Discriminator Loss', 'Discriminator Accuracy'])
plt.show()



plt.plot(G_L)
plt.title('Generator results (MNIST)')
plt.xlabel('Epochs')
plt.ylabel('Generator Loss (blue)')
plt.legend('Generator Loss')
plt.show()


plt.plot(D_L)
plt.title('Discriminator results (CIFAR10)')
plt.xlabel('Epochs')
plt.ylabel('Discriminator Loss (blue), Discriminator Accuracy (orange)')
plt.legend(['Discriminator Loss', 'Discriminator Accuracy'])
plt.show()

plt.plot(G_L)
plt.title('Generator results (CIFAR10)')
plt.xlabel('Epochs')
plt.ylabel('Generator Loss (blue)')
plt.legend('Generator Loss')
plt.show()

