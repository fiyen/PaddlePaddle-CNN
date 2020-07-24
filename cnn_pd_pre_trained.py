"""
查询预训练词典，embedding层为不可训练的预训练的词向量
"""

from paddle import fluid


def cnn_pd(inputs, output_dim, kernel_size=5, dimension=100, conv_filters=40, stride=2, act='relu',
           words_num=10000, use_bias=True, padding_id=0, pre_trained=None):
    if pre_trained is None:
        w_param_attr = fluid.ParamAttr(name='shared_w',
                                       initializer=fluid.initializer.UniformInitializer(),
                                       trainable=True)
    else:
        w_param_attr = fluid.ParamAttr(name='shared_w',
                                       initializer=fluid.initializer.NumpyArrayInitializer(pre_trained),
                                       trainable=False)
    emb = fluid.embedding(inputs, size=[words_num, dimension], is_sparse=True, padding_idx=padding_id,
                          param_attr=w_param_attr)
    print('emb', emb.shape)
    emb = fluid.layers.reshape(emb, shape=[emb.shape[0], 1, emb.shape[1], emb.shape[2]])
    print('emb', emb.shape)
    conv_out = fluid.layers.conv2d(emb, num_filters=conv_filters, stride=(stride, 1),
                                   filter_size=(kernel_size, dimension), act=act, bias_attr=use_bias)
    print('conv_out', conv_out.shape)
    pool = fluid.layers.pool2d(conv_out, pool_size=(2, 1))
    print('pool', pool.shape)
    pred = fluid.layers.fc([pool], size=output_dim, act='softmax')
    return pred


if __name__ == '__main__':
    import data_utils
    import numpy as np
    import sys
    import math
    from lookup import EnVectorizer

    x, y, vocabulary, vocabulary_inv, test_size, _, _ = data_utils.load_data()
    train_x = x[:-test_size]
    train_y = y[:-test_size]
    test_x = x[-test_size:]
    test_y = y[-test_size:]

    # 查找vocabulary_inv中的所有的词向量，然后形成一个词向量矩阵
    dim = 100  # 给定词向量的维度
    pre_trained = False
    if pre_trained:
        vec = EnVectorizer(fast_mode=True, need_pro=True)
        vec.get_path(filename='glove.6B.100d.txt', path='D:/onedrive/work/word_vector', pre_path='D:/test_output')
        vectors = vec.lookup(vocabulary_inv, skip=True)
        vectors = np.asarray(vectors, dtype='float32')
    else:
        vectors = None

    def build_data(x, y, batch_size=64):
        len_x = x.shape[0]
        # shuffle
        shuffle_ix = np.random.permutation(np.arange(len_x))
        x = x[shuffle_ix]
        y = y[shuffle_ix]
        num = int(np.ceil(len_x / batch_size))
        for i in range(num):
            s = i * batch_size
            e = min(len_x, (i + 1) * batch_size)
            ix = np.arange(s, e)
            x_batch = x[ix]
            y_batch = y[ix]
            yield x_batch, y_batch, i + 1, num


    sequence_length = len(train_x[0])
    label_num = train_y[0].shape[0]
    inputs = fluid.data(name='inputs', shape=[None, sequence_length], dtype='int64')
    label = fluid.data(name='label', shape=[None, label_num], dtype='float32')

    main_program = fluid.default_main_program()
    startup_program = fluid.default_startup_program()

    pred = cnn_pd(inputs, output_dim=label_num, kernel_size=3, dimension=dim, conv_filters=50, stride=1, act='relu',
                  words_num=len(vocabulary), use_bias=True, padding_id=0, pre_trained=vectors)
    loss = fluid.layers.cross_entropy(pred, label, soft_label=True)
    ave_loss = fluid.layers.reduce_mean(loss)
    label = fluid.layers.reshape(fluid.layers.argmax(label, axis=1), [-1, 1])
    acc = fluid.layers.accuracy(pred, label)
    emb = main_program.all_parameters()[0]

    test_program = main_program.clone(for_test=True)

    optimizer = fluid.optimizer.Adam(learning_rate=1e-3)
    optimizer.minimize(ave_loss)

    exe = fluid.Executor(fluid.CUDAPlace(0))

    feed_order = ['inputs', 'label']
    feed_var_list_loop = [main_program.global_block().var(var_name) for var_name in feed_order]
    feeder = fluid.DataFeeder(feed_list=feed_var_list_loop, place=fluid.CPUPlace())
    exe.run(startup_program)

    # 训练阶段
    epochs = 5
    history = {'acc': [], 'loss': []}
    for epoch in range(epochs):
        data = build_data(np.array(train_x).astype('int64'), np.array(train_y).astype('float32'), batch_size=32)
        ave_cost = 0.0
        ave_acc = 0.0
        for x_, y_, step, total_step in data:
            avg_cost_np, tem_loss, accuracy = exe.run(main_program, feed=feeder.feed(zip(x_, y_)),
                                                      fetch_list=[ave_loss, loss, acc])
            # print(tem_loss)
            ave_cost = (ave_cost * (float(step) - 1.0) + avg_cost_np[0]) / float(step)
            ave_acc = (ave_acc * (float(step) - 1.0) + accuracy[0]) / float(step)
            if math.isnan(float(avg_cost_np[0])):
                sys.exit("got NaN loss, training failed.")
            if step % 10 == 0 and step < total_step:
                print('epoch: {} - step: {} / {} - acc: {:.4f} - loss: {:.4f}'.format(epoch + 1,
                                                                                      str(step).ljust(
                                                                                          len(str(total_step))),
                                                                                      total_step,
                                                                                      accuracy[0],
                                                                                      avg_cost_np[0]))
            if step == total_step:
                print('epoch: {} done - -ave acc: {:.4f} - ave loss: {:.4f}'.format(epoch + 1, ave_acc, ave_cost))
        history['acc'].append(ave_acc)
        history['loss'].append(ave_cost)

    # 绘图
    import matplotlib.pyplot as plt

    epochs = [i + 1 for i in range(epochs)]
    plt.plot(epochs, history['loss'], 'r-*')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('loss trend')
    plt.show()
    plt.plot(epochs, history['acc'], 'b->')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.title('accuracy trend')
    plt.show()
    # 测试
    test_data = build_data(np.array(test_x).astype('int64'), np.array(test_y).astype('float32'), batch_size=32)
    ave_cost = 0.0
    ave_acc = 0.0
    for x_, y_, step, total_step in test_data:
        avg_cost_np, accuracy = exe.run(test_program, feed=feeder.feed(zip(x_, y_)), fetch_list=[ave_loss, acc],
                                        use_program_cache=True)
        ave_cost = (ave_cost * (float(step) - 1.0) + avg_cost_np[0]) / float(step)
        ave_acc = (ave_acc * (float(step) - 1.0) + accuracy[0]) / float(step)
        if step % 1 == 0 and step < total_step:
            print('step: {} / {} - acc: {:.4f} - loss: {:.4f}'.format(str(step).ljust(len(str(total_step))),
                                                                      total_step,
                                                                      accuracy[0],
                                                                      avg_cost_np[0]))
        if step == total_step:
            print('Done - -ave acc: {:.4f} - ave loss: {:.4f}'.format(ave_acc, ave_cost))
