
import input_data
import tensorflow as tf

with tf.device('/cpu:0'):
    # In[581]:
    mnist = input_data.read_data_sets('./', one_hot=True)
    x = tf.placeholder("float", [None, 784])
    w1_conv2d = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
    b1_conv2d = tf.Variable(tf.truncated_normal([32], stddev=0.1))
    stride1_con2d = [1, 1, 1, 1]
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    h1 = tf.nn.relu(
        tf.nn.conv2d(input=x_image, filter=w1_conv2d, strides=stride1_con2d, padding="SAME") + b1_conv2d)
    h1_1 = tf.nn.max_pool(h1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    print(h1_1)
    #h1_1=tf.reshape(h1_1,[-1,14*14*32])

    filter1_2_conv2d = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
    b2_conv2d = tf.Variable(tf.truncated_normal([64], stddev=0.1))
    h1_2 = tf.nn.relu(
        tf.nn.conv2d(input=h1_1, filter=filter1_2_conv2d, strides=stride1_con2d, padding='SAME') + b2_conv2d)
    h1_3 = tf.nn.max_pool(h1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    keep_pro = tf.placeholder('float')
    h1_3 = tf.nn.relu(tf.reshape(h1_3, [-1, 7 * 7 * 64]))

    #h2 = h1_3  # tf.nn.relu(tf.layers.batch_normalization((tf.matmul(x,w )+b)))
    w2 = tf.Variable(tf.truncated_normal([ 7 * 7 * 64, 1024], stddev=0.1))
    b2 = tf.Variable(tf.truncated_normal([1024], stddev=0.1))
    h3 = tf.nn.dropout(tf.nn.relu((tf.matmul(h1_3, w2) + b2)), keep_pro)
    w3 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
    b3 = tf.Variable(tf.truncated_normal([10], stddev=0.1))
    y = (tf.nn.softmax(tf.nn.sigmoid(tf.matmul(h3, w3) + b3)))
    y_ = tf.placeholder('float', [None, 10])
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy,global_step=global_step)
    equal = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))
    acc = tf.reduce_mean(tf.cast(equal, 'float32'))


#config=tf.ConfigProto(allow_soft_placement=True)
#config.gpu_options.allow_growth=True
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(20000):#2000
        batch_xs,batch_ys=mnist.train.next_batch(50)#300
        #print(batch_xs.shape)
        #print(batch_ys.shape)
        #w2get,w3get=sess.run([w2,w3],feed_dict={x:batch_xs,y_:batch_ys})
        #print('w2get:',w2get[:1])
        #print('w3get:',w3get[:50])
        #yget=sess.run(h1_2,feed_dict={x:batch_xs,y_:batch_ys})
        #print(yget.shape)
        _,acc_get,global_step_=sess.run([train_step,acc,global_step],feed_dict={x:batch_xs,y_:batch_ys,keep_pro:0.5})
        
        #print('yget:',yget[:2])
        #print('w2get:',w2get[:1])
        #print('wget:',wget[:50])
        print('step:(%d)  train:   acc(%lf)'%(global_step_,acc_get))
    #test
    acc_get=sess.run(acc,feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_pro:1})
    print('test:   acc(%lf)'%(acc_get))

