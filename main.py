import skimage
import skimage.io
import skimage.transform
import numpy as np
import tensorflow as tf
import time
from tkinter import filedialog
from tkinter import Tk
from matplotlib import pyplot as plt
import os
import model


def predict_from_pretrained():
    root = Tk()
    path=os.path.abspath("01.input")
    root.filename = filedialog.askopenfilename(initialdir=path, title="choose your file",
                                               filetypes=(("all files", "*.*"), ("png files", "*.png")))
    print("Load image")
    img = skimage.io.imread(root.filename)
    input=img
    img=img.astype(np.float)
    M=img.shape[0]
    N=img.shape[1]
    print("Build Network")
    sess = tf.Session()
    images = tf.placeholder(tf.float32, [1, M, N, 3])
    train_mode = tf.placeholder(tf.bool)
    rime = model.Network('./rime_v_new5_18000.npy', trainable=False)
    rime.build(images)
    sess.run(tf.global_variables_initializer())
    if img.shape == (M, N, 3):
        img = img.reshape((1, M, N, 3))
    x_batch = img / 255.
    print("Test Network")
    _, _ = sess.run([rime.F_out, rime.att], feed_dict={train_mode: False, images: x_batch})

    print("Run Network")
    s = time.time()
    result, att = sess.run([rime.F_out, rime.att], feed_dict={train_mode: False, images: x_batch})
    e = time.time()
    print("Proc. Time:", e - s)
    output_image = np.minimum(np.maximum(result, 0.0), 1)
    result = np.reshape(output_image[0, :, :, :], [M, N, 3])
    image_name=os.path.basename(root.filename)

    print("save result")
    skimage.io.imsave("./02.Results/out_" + image_name, result)

    print("Visualization")
    out=result*255
    out = out.astype(np.uint8)
    region = input.astype(np.float)
    region=region/255
    region[:,:,0]=region[:,:,0]*0.3+att[0, :, :, 0]*0.7
    region[:,:,1]=region[:,:,1]*0.3+(1-att[0, :, :, 0])*0.7
    region[:,:,2]=region[:,:,2]*0.3
    skimage.io.imsave("./02.Results/region_" + image_name, region)
    region = region*255
    visual=np.zeros((M,N*3,3),np.uint8)
    visual[:, 0:N, :]=input
    visual[:, N:N*2, :]=out
    visual[:, N*2:N*3, :]=region.astype(np.uint8)
    skimage.io.imshow(visual)
    plt.show()

if __name__ == "__main__":
    predict_from_pretrained()





