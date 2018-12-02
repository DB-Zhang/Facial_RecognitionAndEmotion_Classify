import os
import cv2 as cv  # Pillow Matplotlib

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt

FLAGS = None


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def conv2d_2(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def max_pool_4x4(x):
    return tf.nn.max_pool(x, ksize=[1, 4, 4, 1],
                          strides=[1, 4, 4, 1], padding='SAME')


def load_pics(cat_dir,dog_dir):
    features = []
    labels = []
    #load cat pics and label them as 0
    list = os.listdir(cat_dir)
    for i in range(0, len(list)):
        file_path = os.path.join(cat_dir, list[i])
        img = cv.imread(file_path,0)
        features.append(img)
        labels.append([0])
    # load dog pics and label them as 1
    list = os.listdir(dog_dir)
    for i in range(0, len(list)):
        file_path = os.path.join(dog_dir, list[i])
        img = cv.imread(file_path,0)
        features.append(img)
        labels.append([1])


    features = np.array(features)
    features = features.reshape(features.shape[0], -1)
    print("features shape {}".format(features.shape))

    labels = np.array(labels)
    labels = labels.reshape(features.shape[0], 1)
    print("labels shape {}".format(labels.shape))

    permutation = np.random.permutation(features.shape[0])
    shuffled_dataset = features[permutation, :]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels


def prepare_data(cat_dir,dog_dir, val_split):
    dataset, labels = load_pics(cat_dir,dog_dir)

    ohe = preprocessing.OneHotEncoder(categorical_features=[0])
    labels = ohe.fit_transform(labels).toarray()

    x_train, x_val, y_train, y_val = train_test_split(dataset, labels, test_size=val_split, random_state=0)

    return x_train, y_train, x_val, y_val


def batch_data(source, target, batch_size):
    for batch_i in range(0, len(source) // batch_size):
        start_i = batch_i * batch_size
        source_batch = source[start_i:start_i + batch_size]
        target_batch = target[start_i:start_i + batch_size]
        # do minmax scale here
        source_batch = preprocessing.minmax_scale(source_batch, axis=1)
        yield source_batch, target_batch


def main(is_train,is_continue):
    train_accuracy = []
    validation_accuracy = []
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    ss = tf.InteractiveSession(config=config)

    train_data, train_label, validate_data, validate_label = prepare_data("Cat_images","Dog_images", 0.2)
    # print(train_data.shape)
    # print(train_label.shape)
    # print(validate_data.shape)
    # print(validate_label.shape)

    x = tf.placeholder(tf.float32, [None, 128 * 128 * 1])
    y_ = tf.placeholder(tf.float32, [None, 2])
    keep_prob = tf.placeholder(tf.float32)

    # 1st layer
    W_conv1 = weight_variable([11, 11, 1, 32])
    b_conv1 = bias_variable([32])

    x_image = tf.reshape(x, [-1, 128, 128, 1])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)  # with pool 2*2, image size from 128*128 -> 64*64

    # 2nd layer
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu6(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)  # with pool 2X2, image size from 64*64 -> 32*32
    # 3rd layer
    W_conv3 = weight_variable([7,7,64,48])
    b_conv3 = bias_variable([48])    
    h_conv3 = tf.nn.relu(conv2d(h_pool2,W_conv3)+b_conv3)
    h_pool3 = max_pool_4x4(h_conv3) # with pool 4X4, image size from 32 -> 8

    W_fc1 = weight_variable([8 * 8 * 48, 128])
    b_fc1 = bias_variable([128])

    h_pool3_flat = tf.reshape(h_pool3, [-1, 8 * 8 * 48])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([128, 32])
    b_fc2 = bias_variable([32])
    h_fc2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    W_fc3 = weight_variable([32, 2])
    b_fc3 = bias_variable([2])
    y = tf.matmul(h_fc2, W_fc3) + b_fc3
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y))

    training_steps = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float32'))
    
    
    saver = tf.train.Saver()
    

    tf.summary.histogram('y', y)
    tf.summary.scalar('loss_function', cross_entropy)
    tf.summary.scalar('accuracy', accuracy)
    merged_summary_op = tf.summary.merge_all()

    if is_train:
        if is_continue:
            saver.restore(ss, "./model")
        else:
            ss.run(tf.global_variables_initializer())
        if tf.gfile.Exists("logs"):
            tf.gfile.DeleteRecursively("logs")
        summary_writer = tf.summary.FileWriter('logs', ss.graph)

        accuracy_line = []
        validate_line = []
        poll_times = 50
        loop = 0
        while loop < poll_times:
            accuracy_total = 0
            batch_run = 0
            batch_size = 64
            training_group = batch_data(train_data, train_label, batch_size)
            validating_group = batch_data(validate_data, validate_label, batch_size)
            for i in range(train_data.shape[0] // batch_size):
                batch_xs, batch_ys = next(training_group)
                ss.run(training_steps, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
                summary_str = ss.run(merged_summary_op, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})
                train_acc = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})
                accuracy_total += train_acc
                batch_run += 1
                print("loop %d, step %d, training accuracy %g" % (loop, i, train_acc))
                summary_writer.add_summary(summary_str, loop * (train_data.shape[0] // batch_size) + i)
            print("accuracy %g" % (accuracy_total * 100 / batch_run))
            train_accuracy.append((accuracy_total * 100 / batch_run))
            accuracy_line.append(accuracy_total * 100 / batch_run)

            validate_acc_total = 0
            validate_batch_run = 0
            for i in range(validate_data.shape[0] // batch_size):
                validate_batch_x, validate_batch_y = next(validating_group)
                validate_acc = accuracy.eval(feed_dict={x: validate_batch_x, y_: validate_batch_y, keep_prob: 1.0})
                validate_batch_run += 1
                validate_acc_total += validate_acc

            print("validate accuracy %g" % (validate_acc_total * 100 / validate_batch_run))
            validate_line.append(validate_acc_total * 100 / validate_batch_run)
            loop += 1

        print(accuracy_line)
        print(validate_line)

        save_path = saver.save(ss, "./model")
        print("Model saved in file: %s" % save_path)

        '''
        train_accuracy = np.array(train_accuracy).reshape((poll_times,1))
        validation_accuracy = np.array(validation_accuracy).reshape((poll_times,1))
        axis = np.arange(1,poll_times,1)
        plt.subplot(2,1,1)
        plt.plot(axis , train_accuracy , '.-',label = "train_accuracy")
        plt.subplot(2,1,2)
        plt.plot(axis , train_accuracy , '-',label = "validation_accuracy")
        '''

    else:
        saver.restore(ss, "./model")
        while True:
            user_input = input("input the index you want to seen\n")  # indicate which figure you want to review
            try:
                index = int(user_input)
            except ValueError:
                if user_input == "" or user_input == "\n":
                    print("user input ends")
                    break
                print("your input is illegal, redo...")
                continue

            candidate = validate_data[index % validate_data.shape[0]]
            # do minmax scale here
            print(candidate)
            candidate_processed = preprocessing.minmax_scale([candidate], axis=1)
            print(candidate_processed)

            result = ss.run(y, feed_dict={x: candidate_processed, keep_prob: 1})
            category = tf.argmax(result, 1)
            category_value = ss.run(category)
            true_result = validate_label[index % validate_data.shape[0]]
            true_category = np.argmax(true_result)
            print('predicted label {} true label {}'.format(category_value[0], true_category))
            cv.imshow("{}".format(index), candidate.reshape(128, 128, 1))
            cv.waitKey(9999)
    ss.close()

    



if __name__ == "__main__":
    is_train = True
    is_continue = False
    main(is_train,is_continue)
