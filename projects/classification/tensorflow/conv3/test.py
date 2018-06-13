import tensorflow as tf
from net import simpleconv3net
import sys
import numpy as np
import cv2

valpath = "all_shuffle_imgs_val.txt"
modelpath = "checkpoints/model.ckpt-9901"

testsize = 48
x = tf.placeholder(tf.float32, [1,testsize,testsize,3])
y = simpleconv3net(x)
y = tf.nn.softmax(y)

sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess,modelpath)


f = open(valpath, "r")
f1 = open('val_wrong.txt' , 'w')
val_num = 0
false_num = 0
false1 = 0
false0 = 0
while 1:
    line = f.readline()
    if line:
        val_num = val_num + 1
        [imgpath, category]=line.split()
        img = cv2.imread(imgpath).astype(np.float32)
        img = cv2.resize(img,(testsize,testsize),interpolation=cv2.INTER_NEAREST)

        imgs = np.zeros([1,testsize,testsize,3],dtype=np.float32)
        imgs[0:1,] = img

        result = sess.run(y, feed_dict={x:imgs})

        if result[0][0]>result[0][1]:
            predict = 0
        else:
            predict = 1

        if int(category)!=predict:
            false_num = false_num + 1
            #print(predict,line)
            f1.write(line+" "+ str(predict) +'\n')
            if int(category)==0:
                false0 = false0 + 1
            else:
                false1 = false1 +1

    else:
        break


print("错误率= %f  ，%d false in %d imgs"%(false_num/val_num, false_num, val_num))
print("不笑的误判个数："+str(false0)+"微笑的误判个数:"+str(false1))
