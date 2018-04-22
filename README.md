# face_recognition
face recognition and face verification system from scratch.


***

## Datasets:
- social face classification(SFC)
- labelled faces in the wild(LFW)
- youtube faces (YTF)

### the model inspired by paper [ going deeper with convolutions]

###  Use a pretrained model to map the images to  feature vectors using triplet loss firstly.
"""
    Implementation of the triplet loss function
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor data
            positive -- the encodings for the positive data (similar to anchor)
            negative -- the encodings for the negative data (different from anchor)
    Returns:
    loss -- real number, value of the loss
"""


'''
def triplet_loss(y_true, y_pred, alpha = 0.4):
    

    anchor = y_pred[:,0:3]
    positive = y_pred[:,3:6]
    negative = y_pred[:,6:9]

    # distance between the anchor and the positive
    pos_dist = K.sum(K.square(anchor-positive),axis=1)

    # distance between the anchor and the negative
    neg_dist = K.sum(K.square(anchor-negative),axis=1)

    # compute loss
    basic_loss = pos_dist-neg_dist+alpha
    loss = K.maximum(basic_loss,0.0)
 
    return loss
'''
"""
    Base network to be shared.
""" 
'''
def create_base_network(in_dims, out_dims):
    
    model = Sequential()
    model.add(BatchNormalization(input_shape=in_dims))
    model.add(LSTM(512, return_sequences=True, dropout=0.2, recurrent_dropout=0.2, implementation=2))
    model.add(LSTM(512, return_sequences=False, dropout=0.2, recurrent_dropout=0.2, implementation=2))
    model.add(BatchNormalization())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(out_dims, activation='linear'))
    model.add(BatchNormalization())

    return model
  
  
in_dims = (N_MINS, n_feat)
out_dims = N_FACTORS
'''

# Network definition

'''
with tf.device(tf_device):
    # Create the 3 inputs
    anchor_in = Input(shape=in_dims)
    pos_in = Input(shape=in_dims)
    neg_in = Input(shape=in_dims)

    # Share base network with the 3 inputs
    base_network = create_base_network(in_dims, out_dims)
    anchor_out = base_network(anchor_in)
    pos_out = base_network(pos_in)
    neg_out = base_network(neg_in)
    merged_vector = concatenate([anchor_out, pos_out, neg_out], axis=-1)

    # Define the trainable model
    model = Model(inputs=[anchor_in, pos_in, neg_in], outputs=merged_vector)
    model.compile(optimizer=Adam(),
                  loss=triplet_loss)

# Training the model
'''
model.fit(train_data, y_dummie, batch_size=256, epochs=10)

'''



