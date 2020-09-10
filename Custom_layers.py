# Define custom layers
def tile_w():
  def tw(w,x):
    new_shape = (int_shape(x)[1],int_shape(x)[2])
    tiled = Lambda(lambda image: tf.image.resize_images(image,new_shape,method = tf.image.ResizeMethod.NEAREST_NEIGHBOR,align_corners = True, preserve_aspect_ratio = True))(w)
    if int_shape(x)[3] == 3:
      tiled = concatenate([tiled,tiled,tiled],axis=-1)
    else:
      for ii in range(int(np.log2(int_shape(x)[3]))):
        tiled = concatenate([tiled,tiled],axis=-1)
    return tiled
  return tw


def weighted_Add():
  def wa(input_1,input_2,w_val):
    #retruns input_1*w + input_2*(1-w)
    w = tile_w()(w_val,input_1)
    one_w = Lambda(lambda x: 1-x)(w)  
    i1_w1 = Multiply()([input_1,w])
    i2_w2 = Multiply()([input_2,one_w])
    added = Add()([i1_w1,i2_w2])
    return added
  return wa

def resize_img(new_shape):
  def res(img_lr_in):
    img_resized =  Lambda(lambda image: tf.image.resize_images(image,new_shape,method = tf.image.ResizeMethod.BICUBIC,align_corners = True, preserve_aspect_ratio = True))(img_lr_in) 
    return img_resized
  return res

def resize_img_NN(new_shape):
  def res(img_lr_in):
    img_resized =  Lambda(lambda image: tf.image.resize_images(image,new_shape,method = tf.image.ResizeMethod.NEAREST_NEIGHBOR,align_corners = True, preserve_aspect_ratio = True))(img_lr_in) 
    return img_resized
  return res

def Dynamic_routing():
  def dr(s_j,uhat):
    c = Activation('softmax')(s_j) 
    c = Dense(160)(c)
    x = Multiply()([uhat, c])
    s_j = LeakyReLU()(x)
    return s_j
  return dr

def Capslayer(casp_num,caps_dim):
  def CP(x):
    x1 = Conv2D(filters=64, kernel_size=9, strides=2, padding='valid', name='primarycap_conv2')(x)
    x2 = Flatten()(x1)
    x22 = Dense(casp_num*caps_dim)(x2)
    x3 = Reshape(target_shape=[-1, caps_dim])(x2)
    x4 = Lambda(squash, name='primarycap_squash')(x3)
    x5 = BatchNormalization(momentum=0.8)(x4)
    x6 = Flatten()(x5)
    return x6
  return CP

def Last_Caps(casp_num,caps_dim):
  def LC(x):
    uhat = Dense(160, kernel_initializer='he_normal', bias_initializer='zeros')(x)
    return uhat
  return LC

def Conv_LR(filters_num=128, kernel_size_num=9, strides_num=2):
  def CL(img):
    x1 = Conv2D(filters=filters_num, kernel_size=kernel_size_num, strides=strides_num, padding='same')(img)
    x11 = LeakyReLU()(x1)
    return x11
  return CL

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

def process_block():
  def pb(input_layer):
    pd = Conv2D(64, kernel_size=3, strides=1, padding='same')(input_layer)
    x1 = residual_block(pd,64)
    x2 = deconv2d(x1,num=64)
    return x2
  return pb
  
'''  
def process_block():
  def pb(input_layer,vgg_features):
    pd = Conv2D(64, kernel_size=3, strides=1, padding='same')(input_layer)
    x1 = residual_block(pd,64)
    x2 = deconv2d(x1,num=64)
    vgg_features_cnn = Conv2D(64, kernel_size=3, strides=1, padding='same')(vgg_features)
    xconcat = concatenate([x2,vgg_features_cnn],axis=-1)
    return xconcat
  return pb
'''

def to_rgb():
  def tr(x):
    x2 = Conv2D(3, kernel_size=3, strides=1, padding='same',activation="tanh")(x)
    return x2
  return tr

def build_resize(in_shape,out_shape):
  img_in = Input(in_shape)
  img_out = Lambda(lambda image: tf.image.resize_images(image,out_shape,method = tf.image.ResizeMethod.NEAREST_NEIGHBOR,align_corners = True, preserve_aspect_ratio = True))(img_in)
  return Model(img_in,img_out)




