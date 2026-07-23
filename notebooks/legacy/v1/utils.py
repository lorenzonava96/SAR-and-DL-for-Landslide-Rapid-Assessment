import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

fil_size1 = 3
fil_size2 = 3
fil_size3 = 3

def CNN_CBAM(lr, loss, filtersFirstLayer, drop, input_size=(64, 64, 4)):
    inputs = Input(shape=input_size)
    conv1 = Conv2D(filtersFirstLayer, fil_size1, padding='same', activation='relu')(inputs)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D()(conv1)

    conv2 = Conv2D(filtersFirstLayer, fil_size2, padding='same', activation='relu')(pool1)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D()(conv2)

    conv3 = Conv2D(filtersFirstLayer, fil_size3, padding='same', activation='relu')(pool2)
    conv3 = BatchNormalization()(conv3)

    target_shape = (conv3.shape[1], conv3.shape[2])

    resized_tensor_3 = tf.image.resize(conv3, target_shape)
    resized_tensor_2 = tf.image.resize(conv2, target_shape)
    resized_tensor_1 = tf.image.resize(conv1, target_shape)

    concatenated_tensor = tf.concat([resized_tensor_3, resized_tensor_3, resized_tensor_3], axis=-1)

    pool3 = MaxPooling2D()(concatenated_tensor)

    drop1 = Dropout(drop)(pool3)
    flat = Flatten()(drop1)
    en = Dense(filtersFirstLayer * 8, activation='relu')(flat)
    out = Dense(1, activation='sigmoid')(en)
    model = Model(inputs, out)

    model.compile(loss=loss, optimizer=Adam(learning_rate=lr), metrics='accuracy')
    return model

def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    # Compute binary cross-entropy
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred, from_logits=True)

    # Compute the predicted probabilities for the true class
    p_t = tf.math.exp(-bce)
    
    # Compute the focal loss
    focal_loss = alpha * (1 - p_t) ** gamma * bce
    
    return focal_loss

def non_max_suppression_fast(boxes, overlapThresh):
    if len(boxes) == 0:
        return []
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    
    pick = []
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
    
    return boxes[pick].astype("int")

def sliding_window(image, step, ws):
    for y in range(0, image.shape[0] - ws[1], step):
        for x in range(0, image.shape[1] - ws[0], step):
            yield (x, y, image[y:y + ws[1], x:x + ws[0]])
