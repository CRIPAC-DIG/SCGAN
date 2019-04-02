import os, sys
sys.path.append(os.getcwd())


import numpy as np

import tensorflow as tf

# import tflib as lib
# import tflib.ops.linear
# import tflib.ops.conv2d
# import tflib.ops.batchnorm
# import tflib.ops.deconv2d


def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)


class WGAN(object):
    def __init__(self, config, mode, device, reuse=True):
        DIM = config.DIM
        BATCH_SIZE = config.BATCH_SIZE
        CRITIC_ITERS = config.CRITIC_ITERS
        LAMBDA = config.LAMBDA
        ITERS = config.ITERS
        GIMG = config.GIMG
        category_num = config.category_num
        D_hidden_layer = config.D_hidden_layer
        learning_rate = config.learning_rate


        with tf.device(device), tf.name_scope(mode), tf.variable_scope("WGAN", reuse=reuse):
            self.input1 = input1 = tf.placeholder(tf.float32, shape=[BATCH_SIZE, GIMG])
            self.input2 = input2 = tf.placeholder(tf.float32, shape=[BATCH_SIZE, GIMG])
            self.ginput = tf.placeholder(tf.float32, shape=[BATCH_SIZE, GIMG])
            self.ginput_c = tf.placeholder(tf.int32, shape=(1,))
            self.input_c1 = input_c1 = tf.placeholder(tf.int32, shape=(1,))
            self.input_c2 = input_c2 = tf.placeholder(tf.int32, shape=(1,))

            self.W_c = tf.get_variable(
                "W_c",
                [category_num, GIMG, DIM],
                initializer=tf.orthogonal_initializer()
            )
            self.D_1 = tf.get_variable(
                "D_1",
                [DIM, D_hidden_layer],
                initializer=tf.orthogonal_initializer()
            )
            self.D_1b = tf.get_variable(
                "D_1b",
                (D_hidden_layer,)
            )
            self.D_2 = tf.get_variable(
                "D_2",
                [D_hidden_layer, 1],
                initializer=tf.orthogonal_initializer()
            )
            gen_params = [self.W_c]
            disc_params = [self.D_1, self.D_1b, self.D_2]
            
            # # parameter setting over

            self.style_1 = style_1 = self.Generator(GIMG, DIM, BATCH_SIZE, input1, input_c1)
            self.style_2 = style_2 = self.Generator(GIMG, DIM, BATCH_SIZE, input2, input_c2)
            W_c_t1 = tf.nn.embedding_lookup(self.W_c, input_c1)
            W_c_t1 = tf.reshape(W_c_t1, [GIMG, DIM])
            W_c_t2 = tf.nn.embedding_lookup(self.W_c, input_c2)
            W_c_t2 = tf.reshape(W_c_t2, [GIMG, DIM])
            self.orth = orthogon = tf.norm(tf.matmul(tf.transpose(W_c_t1), W_c_t1) - tf.ones((DIM, DIM)), 'fro', axis=(0,1)) + \
            tf.norm(tf.matmul(tf.transpose(W_c_t2), W_c_t2) - tf.ones((DIM, DIM)), 'fro', axis=(0,1))


            self.test, disc_real = self.Discriminator(style_1, DIM, D_hidden_layer)
            _, disc_fake = self.Discriminator(style_2, DIM, D_hidden_layer)
            self.disc_real = disc_real
            self.disc_fake = disc_fake
            self.delta = delta = tf.reduce_mean(tf.nn.moments(style_1, 0)[1] + tf.nn.moments(style_2, 0)[1]) - \
            tf.reduce_mean(tf.nn.moments(input1, 0)[1] + tf.nn.moments(input2, 0)[1])
            self.delta1 = tf.reduce_mean(tf.nn.moments(style_1, 0)[1] + tf.nn.moments(style_2, 0)[1])
            self.delta2 = tf.reduce_mean(tf.nn.moments(input1, 0)[1] + tf.nn.moments(input2, 0)[1])

            # self.delt1 =  tf.reduce_sum(tf.nn.moments(style_1, 0)[1] + tf.nn.moments(style_2, 0)[1])

            # self.gen_cost = gen_cost =  tf.abs(tf.reduce_mean(disc_real) - tf.reduce_mean(disc_fake)) #+ config.delta * tf.abs(delta)  #/ config.BATCH_SIZE
            # self.disc_cost = disc_cost = -tf.abs(tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)) # / config.BATCH_SIZE
            self.gen_cost = gen_cost =  tf.reduce_mean(disc_real) - tf.reduce_mean(disc_fake) #+ config.delta * tf.abs(delta)  #/ config.BATCH_SIZE
            self.disc_cost = disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real) # / config.BATCH_SIZE
    
            gen_cost += config.delta * orthogon
            # gen_cost = config.delta * delta
            # alpha = tf.random_uniform(
            #     shape=[BATCH_SIZE, 1],
            #     minval=0.,
            #     maxval=1.
            # )
            # differences = style_1 - style_2
            # interpolates = style_2 + (alpha * differences)
            # gradients = tf.gradients(self.Discriminator(interpolates, DIM, D_hidden_layer)[1], [interpolates])[0]
            # slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            # gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
            # disc_cost += LAMBDA * gradient_penalty

            self.clip_disc_weights = None

        if mode == "Train":
            # self.gen_train_op = tf.train.AdamOptimizer(
            #     learning_rate= 1e-4,
            #     beta1=0.5,
            #     beta2=0.9
            # ).minimize(gen_cost, var_list=gen_params)
            # self.disc_train_op = tf.train.AdamOptimizer(
            #     learning_rate= 1e-4,
            #     beta1=0.5,
            #     beta2=0.9
            # ).minimize(disc_cost, var_list=disc_params)
            # optimizer = tf.train.AdamOptimizer(learning_rate= 1e-3, beta1=0.5, beta2=0.9)
            optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
            gen_grad = tf.gradients(gen_cost, gen_params)
            disc_grad = tf.gradients(disc_cost, disc_params)
            gen_grad, _ = tf.clip_by_global_norm(gen_grad, config.max_grad_norm)
            disc_grad, _ = tf.clip_by_global_norm(disc_grad, config.max_grad_norm)
            self.gen_train_op = optimizer.apply_gradients(
                zip(gen_grad, gen_params),
                global_step=tf.contrib.framework.get_or_create_global_step()
            )
            self.disc_train_op = optimizer.apply_gradients(
                zip(disc_grad, disc_params),
                global_step=tf.contrib.framework.get_or_create_global_step()
            )

            # self.clip_disc_weights = None
        else:
            self.gen_train_op = tf.no_op()
            self.disc_train_op = tf.no_op()
            self.goutput = self.Generator(GIMG, DIM, BATCH_SIZE, self.ginput, self.ginput_c)

    def Generator(self, IMG_DIM, DIM, n_samples, inputvector, category):
        # if noise is None:
        #     noise = tf.random_normal([n_samples, 128])
        inputvector = tf.reshape(inputvector, [n_samples, IMG_DIM])
        W_c_t = tf.nn.embedding_lookup(self.W_c, category)
        W_c_t = tf.reshape(W_c_t, [IMG_DIM, DIM])
        output = tf.matmul(inputvector, W_c_t)
        # output = tf.nn.relu(output)
        # output = tf.sigmoid(output)
        # print output
        return tf.reshape(output, [-1, DIM])

    def Discriminator(self, inputs, DIM, D_hidden_layer):
        output = tf.reshape(inputs, [-1, DIM])
        # print output
        output = tf.matmul(output, self.D_1) + self.D_1b
        output = tf.sigmoid(output)
        output = tf.matmul(output, self.D_2)
        # output = tf.nn.relu(output)
        test = 0
        return test, tf.reshape(output, [-1])


