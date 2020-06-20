from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib as mpl

files = sys.argv[1:]
net = load_model('model.h5')
cls_list = ['cat', 'dog']

pretrained_model = tf.keras.models.load_model('model.h5')
pretrained_model.trainable = False
loss_object = tf.keras.losses.CategoricalCrossentropy()

def preprocess(image):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    return image

def save_images(image):
    img = plt.imshow(image[0]*0.5+0.5, interpolation='nearest')
    img.set_cmap('hot')
    plt.axis('off')
    plt.savefig('adversarial_' + f, bbox_inches = 'tight', pad_inches = 0)

def create_adversarial_pattern(input_image, input_label):
    input_label = label
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = pretrained_model(input_image)
        loss = loss_object(input_label, prediction)

    # Get the gradients of the loss w.r.t to the input image.
    gradient = tape.gradient(loss, input_image)
    # Get the sign of the gradients to create the perturbation
    return tf.sign(gradient)

if __name__ == '__main__':
    for f in files:
        img = image.load_img(f, target_size=(224, 224))
        if img is None:
            continue
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis = 0)
        image_probs = net.predict(x)
        pred = image_probs[0]
        top_inds = pred.argsort()[::-1][:5]

        print(f,' is ',cls_list[top_inds[0]])
        for i in top_inds:
            print('    {:.3f}  {}'.format(pred[i], cls_list[i]))
        print()

        # Get the input label of the image.
        labrador_retriever_index = top_inds[0]
        label = tf.one_hot(labrador_retriever_index, image_probs.shape[-1])
        label = tf.reshape(label, (1, image_probs.shape[-1]))
        input_image = tf.convert_to_tensor(x, dtype=tf.float32)

        perturbations = create_adversarial_pattern(input_image, label)
        #plt.imshow(perturbations[0]*0.5+0.5); # To change [-1, 1] to [0,1]
        #plt.show()

        epsilons = [1]

        for i, eps in enumerate(epsilons):
            adv_x = input_image + eps*perturbations
            pred = net.predict(adv_x, steps=10)[0]
            top_inds = pred.argsort()[::-1][:5]
            print('adversarial_' + f,' is ',cls_list[top_inds[0]])
            for i in top_inds:
                print('    {:.3f}  {}'.format(pred[i], cls_list[i]))
            adv_x = preprocess(adv_x)
            adv_x = tf.clip_by_value(adv_x, -1, 1)
            save_images(adv_x)
