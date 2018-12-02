from datetime import datetime
import tensorflow as tf
import pandas as pd
import numpy as np
import csv
import cv2 as cv
import os
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt

Angry_train = []
Happy_train = []
Angry_val = []
Happy_val = []


def get_img(pic_path):
	pic = cv.imread(pic_path,0)
	return pic


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


def load_pics(Angry_dir ,Happy_dir,Others_dir):
    features = []
    labels = []
    #load Angry_dir and label them as 0
    list = os.listdir(Angry_dir)
    for i in range(0, len(list)):
        file_path = os.path.join(Angry_dir, list[i])
        img = cv.imread(file_path,0)
        features.append(img)
        labels.append([0])


    # load Happy_dir and label them as 1
    list = os.listdir(Happy_dir)
    for i in range(0, len(list)):
        file_path = os.path.join(Happy_dir, list[i])
        img = cv.imread(file_path,0)
        features.append(img)
        labels.append([1])

    # load Others_dir pics and label them as 2
    list = os.listdir(Others_dir)
    for i in range(0, len(list)):
        file_path = os.path.join(Others_dir, list[i])
        img = cv.imread(file_path,0)
        features.append(img)
        labels.append([2])


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


def get_dataset(Angry_dir ,Happy_dir,Others_dir):
    dataset, labels = load_pics(Angry_dir,Happy_dir,Others_dir)

    ohe = preprocessing.OneHotEncoder(categorical_features=[0])
    labels = ohe.fit_transform(labels).toarray()

    X, _, Y, _= train_test_split(dataset, labels, test_size=0, random_state=0)

    return X, Y


def batch_data(source, target, batch_size):
    for batch_i in range(0, len(source) // batch_size):
        start_i = batch_i * batch_size
        source_batch = source[start_i:start_i + batch_size]
        target_batch = target[start_i:start_i + batch_size]
        # do minmax scale here
        source_batch = preprocessing.minmax_scale(source_batch, axis=1)
        yield source_batch, target_batch


def Classify_train(is_train,is_continue,looptimes=20):
    train_accuracy = []
    validation_accuracy = []
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    ss = tf.InteractiveSession(config=config)
    train_data, train_label = get_dataset("Training/Angry","Training/Happy","Training/Others")
    validate_data, validate_label = get_dataset("PublicTest/Angry","PublicTest/Happy","PublicTest/Others")
    x = tf.placeholder(tf.float32, [None, 48 * 48 * 1])
    y_ = tf.placeholder(tf.float32, [None, 3])
    keep_prob = tf.placeholder(tf.float32)

    x_image = tf.reshape(x, [-1, 48, 48, 1])
    # 1st layer
    W_conv1_1 = weight_variable([3, 3, 1, 64])
    b_conv1_1 = bias_variable([64])
    h_conv1_1 = tf.nn.relu(conv2d(x_image, W_conv1_1) + b_conv1_1)


    W_conv1_2 = weight_variable([3, 3, 64, 64])
    b_conv1_2 = bias_variable([64])
    h_conv1_2 = tf.nn.relu(conv2d(h_conv1_1, W_conv1_2) + b_conv1_2)

    h_pool1 = max_pool_2x2(h_conv1_2)  # with pool 2*2, image size from 48*48 -> 24*24

    # 2nd layer
    W_conv2_1 = weight_variable([3, 3, 64, 128])
    b_conv2_1 = bias_variable([128])
    h_conv2_1 = tf.nn.relu6(conv2d(h_pool1, W_conv2_1) + b_conv2_1)

    W_conv2_2 = weight_variable([3, 3, 128, 128])
    b_conv2_2 = bias_variable([128])
    h_conv2_2 = tf.nn.relu6(conv2d(h_conv2_1, W_conv2_2) + b_conv2_2)

    h_pool2 = max_pool_2x2(h_conv2_2)  # with pool 2X2, image size from 24*24 -> 12*12


    # 3rd layer
    W_conv3_1 = weight_variable([3,3,128,256])
    b_conv3_1 = bias_variable([256])    
    h_conv3_1 = tf.nn.relu(conv2d(h_pool2,W_conv3_1)+b_conv3_1)

    W_conv3_2 = weight_variable([3,3,256,256])
    b_conv3_2 = bias_variable([256])    
    h_conv3_2 = tf.nn.relu(conv2d(h_conv3_1,W_conv3_2)+b_conv3_2)

    h_pool3 = max_pool_2x2(h_conv3_2) # with pool 2X2, image size from 12*12 -> 6*6

    W_fc1 = weight_variable([6 * 6 * 256, 1024])
    b_fc1 = bias_variable([1024])

    h_pool3_flat = tf.reshape(h_pool3, [-1, 6 * 6 * 256])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 1024])
    b_fc2 = bias_variable([1024])
    h_fc2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    W_fc3 = weight_variable([1024, 3])
    b_fc3 = bias_variable([3])
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
        poll_times = looptimes
        loop = 0
        while loop < poll_times:
            accuracy_total = 0
            batch_run = 0
            batch_size = 32
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
            save_path = saver.save(ss, "./model")

        print(accuracy_line)
        print(validate_line)

        save_path = saver.save(ss, "./model")
        tf.add_to_collection('pred_network',y)
        print("Model saved in file: %s" % save_path)

def predict_pic(pic_path):
    '''pic_data = []
    pic_data.append(get_img(pic_path))
    pic_data = np.array(pic_data)
    print("pic_data shape {}".format(pic_data.shape))
    pic_data.reshape(pic_data.shape[0], -1)
    np.reshape(pic_data,(1,2304))
    print("pic_data shape {}".format(pic_data.shape))'''
    pic_data = np.array(get_img(pic_path))
    X = np.reshape(pic_data,(1,2304))
    print("pic_data shape {}".format(pic_data.shape))
    #x_data = pic_data[0]
    #x_data_progress = preprocessing.minmax_scale([x_data], axis=1)

    print(pic_data.shape)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    ss = tf.InteractiveSession(config=config)

    x = tf.placeholder(tf.float32, [None, 48 * 48 * 1])
    keep_prob = tf.placeholder(tf.float32)

    x_image = tf.reshape(x, [-1, 48, 48, 1])
    # 1st layer
    W_conv1_1 = weight_variable([3, 3, 1, 64])
    b_conv1_1 = bias_variable([64])
    h_conv1_1 = tf.nn.relu(conv2d(x_image, W_conv1_1) + b_conv1_1)


    W_conv1_2 = weight_variable([3, 3, 64, 64])
    b_conv1_2 = bias_variable([64])
    h_conv1_2 = tf.nn.relu(conv2d(h_conv1_1, W_conv1_2) + b_conv1_2)

    h_pool1 = max_pool_2x2(h_conv1_2)  # with pool 2*2, image size from 48*48 -> 24*24

    # 2nd layer
    W_conv2_1 = weight_variable([3, 3, 64, 128])
    b_conv2_1 = bias_variable([128])
    h_conv2_1 = tf.nn.relu6(conv2d(h_pool1, W_conv2_1) + b_conv2_1)

    W_conv2_2 = weight_variable([3, 3, 128, 128])
    b_conv2_2 = bias_variable([128])
    h_conv2_2 = tf.nn.relu6(conv2d(h_conv2_1, W_conv2_2) + b_conv2_2)

    h_pool2 = max_pool_2x2(h_conv2_2)  # with pool 2X2, image size from 24*24 -> 12*12


    # 3rd layer
    W_conv3_1 = weight_variable([3,3,128,256])
    b_conv3_1 = bias_variable([256])    
    h_conv3_1 = tf.nn.relu(conv2d(h_pool2,W_conv3_1)+b_conv3_1)

    W_conv3_2 = weight_variable([3,3,256,256])
    b_conv3_2 = bias_variable([256])    
    h_conv3_2 = tf.nn.relu(conv2d(h_conv3_1,W_conv3_2)+b_conv3_2)

    h_pool3 = max_pool_2x2(h_conv3_2) # with pool 2X2, image size from 12*12 -> 6*6

    W_fc1 = weight_variable([6 * 6 * 256, 1024])
    b_fc1 = bias_variable([1024])

    h_pool3_flat = tf.reshape(h_pool3, [-1, 6 * 6 * 256])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 1024])
    b_fc2 = bias_variable([1024])
    h_fc2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    W_fc3 = weight_variable([1024, 3])
    b_fc3 = bias_variable([3])
    y = tf.matmul(h_fc2, W_fc3) + b_fc3
    
    saver = tf.train.Saver()
    saver.restore(ss, "./model")

    result = ss.run(y, feed_dict={x: X, keep_prob: 1})
    predict_result = tf.argmax(result, 1)
    predict_value = ss.run(predict_result)
    if (predict_value == 0):
        emotion = 'Angry'
        print('Angry')
    elif (predict_value ==1):
        emotion = 'Happy'
        print('Happy')
    else:
        emotion = 'Others'
        print('Others')

    ss.close()
    return emotion

if __name__ == "__main__":
    is_train = True
    is_continue = False
    Classify_train(is_train,is_continue,50)
    #predict_pic('1.jpg')
