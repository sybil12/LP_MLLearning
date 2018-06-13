from dataset import *
from net import simpleconv3net
import sys
import cv2

txtfile = sys.argv[1]
batch_size = 64
num_classes = 2
image_size = (48, 48)
learning_rate = 0.0001

debug = False

if __name__ == "__main__":
    dataset = ImageData(txtfile, batch_size, num_classes, image_size)
    iterator = dataset.data.make_one_shot_iterator()
    dataset_size = dataset.dataset_size
    batch_images, batch_labels = iterator.get_next()
    Ylogits = simpleconv3net(batch_images)

    print("Ylogits size=", Ylogits.shape)

    Y = tf.nn.softmax(Ylogits)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=batch_labels)
    cross_entropy = tf.reduce_mean(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(batch_labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

    # add summary
    cost_summary = tf.summary.scalar(cross_entropy.op.name,cross_entropy)
    accuracy_summary = tf.summary.scalar(accuracy.op.name,accuracy)
    learning_rate_summary = tf.summary.scalar(learning_rate.op.name, learning_rate)

    saver = tf.train.Saver()
    in_steps = 100
    checkpoint_dir = 'checkpoints/'

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        #tf.image_summary("x", x_new, max_images=1)
        merged_summary_op = tf.summary.merge_all()  # 合并所有summary nodes为一个操作
        summary_writer =  tf.summary.FileWriter("E://1Software//TensorBoard//test",sess.graph)  #

        steps = 10000
        for i in range(steps+1):
            _, cross_entropy_, accuracy_, batch_images_, batch_labels_, summary_ = sess.run(
                [train_step, cross_entropy, accuracy, batch_images, batch_labels,merged_summary_op])
            summary_writer.add_summary(summary_)  #把统计的op的值写入指定目录

            if i % in_steps == 0:
                print(i, "iterations,loss=", cross_entropy_, "acc=", accuracy_)
                saver.save(sess, checkpoint_dir + 'model.ckpt', global_step=i + 1)
                # print "predict=",Ylogits," labels=",batch_labels

                if debug:
                    imagedebug = batch_images_[0].copy()
                    imagedebug = np.squeeze(imagedebug)
                    print(imagedebug, imagedebug.shape)
                    print(np.max(imagedebug))
                    imagelabel = batch_labels_[0].copy()
                    print(np.squeeze(imagelabel))

                    imagedebug = cv2.cvtColor((imagedebug * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                    cv2.namedWindow("debug image", 0)
                    cv2.imshow("debug image", imagedebug)
                    k = cv2.waitKey(0)
                    if k == ord('q'):
                        break
