from numpy.random import seed

# seed(333)
import tensorflow as tf

# tf.random.set_seed(333)

from tensorflow.keras import optimizers

opt = optimizers.Adadelta(lr=0.0001, rho=0.9, epsilon=1e-07, name="Adadelta")
#opt = optimizers.Adam(lr=0.00001, name="Adam")


#opt = optimizers.RMSprop(lr=0.00003, rho=0.9, epsilon=1e-07, name="rmsprop")
#opt = optimizers.SGD(lr=0.000001, momentum=0.9, nesterov=True, name="SGD")


def network(inp_dim):
    from tensorflow import keras

    model = keras.Sequential()

    # Convolutional layer and maxpool layer 1
    model.add(keras.layers.Conv2D(8, (3, 3), activation='relu', input_shape=(inp_dim[0], inp_dim[1], inp_dim[2])))
    # model.add(keras.layers.MaxPool2D(2, 2))
    model.add(keras.layers.AveragePooling2D(2, 2))

    # model.add(keras.layers.Dropout(0.3))

    # Convolutional layer and maxpool layer 2
    model.add(keras.layers.Conv2D(8, (3, 3), activation='relu'))
    # model.add(keras.layers.MaxPool2D(2, 2))
    model.add(keras.layers.AveragePooling2D(2, 2))

    # model.add(keras.layers.Dropout(0.3))

    # Convolutional layer and maxpool layer 3
    model.add(keras.layers.Conv2D(8, (3, 3), activation='relu'))
    model.add(keras.layers.AveragePooling2D(2, 2))

    model.add(keras.layers.Conv2D(8, (3, 3), activation='relu'))
    model.add(keras.layers.AveragePooling2D(2, 2))

    # model.add(keras.layers.Dropout(0.1))

    # Convolutional layer and maxpool layer 4
    # model.add(keras.layers.Conv2D(4, (3, 3), activation='relu'))
    # model.add(keras.layers.AveragePooling2D(2, 2))

    # model.add(keras.layers.Dropout(0.1))

    # Convolutional layer and maxpool layer 5
    # model.add(keras.layers.Conv2D(4, (3, 3), activation='relu'))
    # model.add(keras.layers.AveragePooling2D(2, 2))

    # Convolutional layer and maxpool layer 4

    # This layer flattens the resulting image array to 1D array
    model.add(keras.layers.Flatten())

    # Hidden layer with 512 neurons and Rectified Linear Unit activation function
    # model.add(keras.layers.Dense(2592, activation='relu'))

    # model.add(keras.layers.Dropout(0.1))

    # Output layer with single neuron which gives 0 for Cat or 1 for Dog
    # Here we use sigmoid activation function which makes our model output to lie between 0 and 1
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    #model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['binary_accuracy'])
    # model.compile(optimizer=opt, loss=contrastive_loss(model, loss_tmp_t, loss_tmp_v), metrics=['binary_accuracy'])
    model.compile(optimizer=opt, loss=contrastive_loss, metrics=[contrastive_acc])
    model.summary()
    return model


def get_compatible_label(label):
    # print(label)
    # print(os.path.basename(label))
    if os.path.basename(label) == 'Cat':
        return 0.
    else:
        return 1.


def get_another_file_from_category(category, current_filename):
    global files_t
    # get a different file from same category, to estimate and average out train loss on same training sample as well as different sample of same category
    same_cat = category
    files_in_same_cat = os.listdir(dataset_path + "/training_set/" + same_cat)

    # Find a unique file from same category
    rnd = files_in_same_cat.index(current_filename)

    while current_filename == os.path.basename(files_in_same_cat[rnd]):
        rnd = random.randint(0, len(files_in_same_cat) - 1)

    print("cur dir: {},  cur file: {}, rnd file: {}".format(category, current_filename,
                                                            os.path.basename(files_in_same_cat[rnd])))

    img_j = np.array(
        image.load_img(os.path.join(dataset_path + "/training_set/" + same_cat, files_in_same_cat[rnd]),
                       target_size=input_dims,
                       grayscale=False)) / 255

    return img_j


def contrastive_loss(y_true, y_pred):
    import tensorflow as tf

    import tensorflow.keras.backend as K



    #return K.mean((1 - y_true) * (1/2) * K.abs(K.square(y_pred)) + ((y_true) * (1/2) * K.abs(K.square(1 - K.mean(y_pred)))))
    return K.mean(K.square(((1 - y_true) * y_pred)) + K.square((y_true) *  (1 - y_pred)))

    #return loss


def contrastive_acc(y_true, y_pred):
    import tensorflow.keras.backend as K
    from tensorflow.python.ops import math_ops
    '''
    cat_img_path = '/mnt/sda1/research/datasets/kagglecatsanddogs_3367a/test_set/Cat/12099.jpg'
    dog_img_path = '/mnt/sda1/research/datasets/kagglecatsanddogs_3367a/test_set/Dog/12157.jpg'

    img_cat = np.array(image.load_img(cat_img_path, target_size=input_dims, grayscale=False)) / 255
    img_dog = np.array(image.load_img(dog_img_path, target_size=input_dims, grayscale=False)) / 255

    emb_cat = model.predict(np.array([img_cat]))[0]
    emb_dog = model.predict(np.array([img_dog]))[0]
    '''


    #return 1-K.mean(((1 - y_true) * y_pred) + ((y_true) * (1 - y_pred)))
    diff = (((1-y_true) * (math_ops.cast(y_pred < 0.2, y_pred.dtype))) + ((y_true) * (math_ops.cast(y_pred >= 0.8, y_pred.dtype))))
    return K.mean(math_ops.equal(diff, 1), axis=-1)

working = True


def press(key):
    global working
    try:
        if key.char == 'q':
            working = False
    except AttributeError:
        pass


def release(key):
    pass
    # print(type(key))
    # if key == 'q':
    #     working = False
    #     print('Azam')
    # if key == Key.space:
    #     return False  # Returns False to stop the listener


if __name__ == '__main__':
    import os
    import sys
    from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau

    from tensorflow.keras.preprocessing import image
    import gc, random
    import numpy as np
    from pynput.keyboard import Key, Listener

    import cv2

    work_dir = os.path.abspath('/mnt/sda2/predator/research/saved_models/')
    dataset_path = os.path.abspath("/mnt/sda1/research/datasets/kagglecatsanddogs_3367a/")
    input_dims = (150, 150, 3)

    x_train = []  # Training set array of compatible data (images)
    y_train = []  # Training set array of compatible data (Labels)

    x_val = []  # Validation set array of compatible data (images)
    y_val = []  # Validation set array of compatible data (Labels)

    loss_th = 0.25
    vloss_th = 0.3
    acc_th = 0.7
    vacc_th = 0.7
    v = 0

    task = 10

    epochs = 10

    categories_training = [os.path.join(dataset_path + "/training_set", o) for o in
                           os.listdir(dataset_path + "/training_set")
                           if os.path.isdir(os.path.join(dataset_path + "/training_set", o))]
    # categories_training.sort()

    files_t = []
    labels_t = []

    share = int(task / 2)

    for idx, cat in enumerate(categories_training):
        tmp = os.listdir(os.path.join(dataset_path + "/training_set", cat))
        if len(tmp) > share:
            tmp = tmp[:share]
        files_t.extend(tmp)
        labels_t.extend(os.path.basename(cat) for d in tmp)

        if len(files_t) >= task / 2:
            continue

    print("Training labels: {}".format(labels_t))
    # Validation Data generation
    categories_validation = [os.path.join(dataset_path + "/validation_set", o) for o in
                             os.listdir(dataset_path + "/validation_set")
                             if os.path.isdir(os.path.join(dataset_path + "/validation_set", o))]
    # categories_validation.sort()

    files_v = []
    labels_v = []

    for cat in categories_validation:
        # print("val cat: {}".format(cat))
        tmp = os.listdir(os.path.join(dataset_path + "/validation_set", cat))
        if len(tmp) > 50:
            tmp = tmp[:50]
        files_v.extend(tmp)
        labels_v.extend(os.path.basename(cat) for d in tmp)

        if len(files_v) >= 50:
            continue

    print("Validation labels: {}".format(labels_v))
    print("number of categories in training set: {} and in validation set: {}".format(len(categories_training),
                                                                                      len(categories_validation)))
    print("number of files in training set: {} and in validation set: {}".format(len(files_t), len(files_v)))

    # ---------------------------------------------------------------
    v = 0



    model = network(input_dims)

    for i in range(len(files_v)):
        img_i = np.array(image.load_img(os.path.join(dataset_path + "/validation_set/" + labels_v[i], files_v[i]),
                                        target_size=input_dims, grayscale=False)) / 255

        x_val.append(img_i)  # True sample
        y_val.append(get_compatible_label(labels_v[i]))

        if len(x_val) >= 50:
            break
        # cv2.waitKey()

    c = list(zip(x_val, y_val))

    random.shuffle(c)

    x_val, y_val = zip(*c)
    # ------------------------------------------------------------------------------
    # Parameters for loss function

    # model.load_weights(os.path.join(work_dir, 'train4.py.loss=4.018052277388051e-05,acc=1.0,vl=0.020762890577316284,va=0.9433962106704712.h5'))
    # chk = ModelCheckpoint(work_dir + '/' + os.path.basename(sys.argv[0]) + '.ep={epoch:02d},loss={loss:.6f},acc={accuracy:.6f},vl={val_loss:.6f},va={val_accuracy:.6f}.h5',   save_weights_only=True, monitor='val_loss', save_best_only=True, period=10)
    csv_logger = CSVLogger(work_dir + '/' + os.path.basename(sys.argv[0]) + '_training2.log')

    # br = tf.keras.callbacks.experimental.BackupAndRestore(os.path.join(work_dir + '/backup/', os.path.basename(sys.argv[0])))

    cb = [csv_logger]

    min_val_loss = 1

    # ...or, in a non-blocking fashion:
    listener = Listener(
        on_press=press,
        on_release=release)
    listener.start()

    end = False

    while not end:
        end_for_loop = False
        for i in range(0, len(files_t) - 1):
            if len(y_train) > task:
                end = True
                break
            if i > task / 2:
                loss_th = 0.15
                vloss_th = 0.2
                acc_th = 0.8
                vacc_th = 0.8
            print(f"i={i}")

            # -------------------------------------------------------------------
            # ------------- CONVERTING DATASET TO COMPATIBLE FORMAT -------------
            # -------------------------------------------------------------------

            same_cat = dataset_path + "/training_set/" + labels_t[random.randint(0, len(labels_t)-1)]

            files_in_same_cat = os.listdir(same_cat)
            # print("files in diff cat: {}".format(files_in_diff_cat))

            rnd = random.randint(0, len(files_in_same_cat) - 1)
            while True:

                if os.path.exists(os.path.join(dataset_path + "/training_set/" + labels_t[i], files_in_same_cat[rnd])):
                    img_i = np.array(image.load_img(os.path.join(dataset_path + "/training_set/" + labels_t[i], files_in_same_cat[rnd]),
                                            target_size=input_dims, grayscale=False)) / 255
                    break
                else:
                    rnd = random.randint(0, len(files_in_same_cat) - 1)

            x_train.append(img_i)  # True sample
            y_train.append(get_compatible_label(labels_t[i]))

            #cv2.imshow('1', img_i)
            #print("{} - res: {}".format(files_in_same_cat[rnd], y_train[-1]))


            diff_cat = dataset_path + "/training_set/" + labels_t[i]
            while diff_cat == (dataset_path + "/training_set/" + labels_t[i]):
                    diff_cat = dataset_path + "/training_set/" + labels_t[random.randint(0, len(labels_t) - 1)]
            #print("current cat: {}, diff cat: {}".format(labels_t[i], os.path.basename(diff_cat)))

            files_in_diff_cat = os.listdir(diff_cat)
            # print("files in diff cat: {}".format(files_in_diff_cat))

            rnd = random.randint(0, len(files_in_diff_cat) - 1)
            img_j = np.array(image.load_img(os.path.join(diff_cat, files_in_diff_cat[rnd]), target_size=input_dims,
                                            grayscale=False)) / 255

            x_train.append(img_j)
            y_train.append(get_compatible_label(os.path.basename(diff_cat)))

            #cv2.imshow('2', img_j)
            #print("{} - res: {}".format(files_in_diff_cat[rnd], y_train[-1]))

            #cv2.waitKey()
            # print("z_train: {}".format(z_train))
            # Reset loss and accuracy values after every new sample of data is added
            loss = 1
            vloss = 1
            acc = 0

            count = 0
            prev_loss = 0
            prev_vacc = 0
            # batch = len(y_train)
            batch = 128
            if batch > 128:
                batch = 128

            # test_loss, test_acc = model.evaluate(np.array(y_train), np.array(z_train), batch_size=batch)
            # test_vloss, test_vacc = model.evaluate(np.array(x_val), np.array(y_val), batch_size=batch)
            # if (test_loss < loss_th) and (test_acc > acc_th) and (test_vloss < vloss_th) and (test_vacc > vacc_th):
            #    continue

            #print("training set labels: {}".format(y_train))
            #print("validation set labels: {}".format(y_val))
            #print("x_val: {}".format(len(x_val)))
            '''
            for img_idx, img in enumerate(y_train):
                print(z_train[img_idx])
                cv2.imshow('train', img)
                cv2.waitKey()

            for img_idx, img in enumerate(x_val):
                print(y_val[img_idx])
                cv2.imshow('val', img)
                cv2.waitKey()
            '''
            '''
            for iidx, img in enumerate(loss_tmp_t):
                cv2.imshow('1', img)
                cv2.imshow('2', y_train[iidx])
                print(z_train[iidx])
                cv2.waitKey()

            for iidx, img in enumerate(loss_tmp_v):
                cv2.imshow('v1', img)
                cv2.imshow('v2', x_val[iidx])
                print(y_val[iidx])
                cv2.waitKey()
            '''
            while True:  # for embeddings under control
                print(
                    "Training on {} sample pairs and validating on {} sample pairs".format(
                        len(x_train) / 2,
                        len(x_val) / 2))
                #print("size of val labels: {}".format(len(y_val)))
                h = model.fit(np.array(x_train), np.array(y_train), batch_size=batch, epochs=epochs, verbose=0,
                              validation_data=(np.array(x_val), np.array(y_val)), callbacks=cb, shuffle=False,
                              steps_per_epoch=None, use_multiprocessing=True, validation_steps=None)
                if not working:
                    working = True
                    break
                # if count >= 10:
                #    break
                loss = h.history['loss'][-1]
                vloss = h.history['val_loss'][-1]
                #acc = h.history['accuracy'][-1]
                acc = h.history['contrastive_acc'][-1]
                #acc = h.history['binary_accuracy'][-1]
                #vacc = h.history['val_accuracy'][-1]
                vacc = h.history['val_contrastive_acc'][-1]
                #vacc = h.history['val_binary_accuracy'][-1]

                print("loss: {}, acc:{}, count: {}, val_acc: {}, val_loss: {}".format(loss, acc, count, vacc, vloss))

                if min_val_loss > vloss:
                    min_val_loss = vloss

                    # save checkpoint
                    model.save(
                        work_dir + '/' + os.path.basename(sys.argv[0]) + '.loss={},acc={},vl={},va={}.h5'.format(loss,
                                                                                                                 acc,
                                                                                                                 vloss,
                                                                                                                 vacc))

                if (prev_loss <= vloss) and (prev_vacc >= vacc):
                    count = count + 1
                    if not((loss > loss_th) or (vloss > vloss_th) or (acc < acc_th) or (vacc < vacc_th)):
                        break
                    if acc > acc_th and vacc > vacc_th:
                        break
                else:
                    count = 0
                    if acc > acc_th and vacc > vacc_th:
                        break
                if count >= 20:


                    #model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['binary_accuracy'])
                    model.compile(optimizer=opt, loss=contrastive_loss, metrics=[contrastive_acc])
                    count = 0
                    i = 0
                    x_train = []
                    y_train = []

                    end_for_loop = True
                    break

                #    break
                # if loss < 0.05 and acc > 0.9 and count > 5 and prev_loss <= vloss:
                #    break


                prev_loss = vloss
                prev_vacc = vacc
            gc.collect()
            if end_for_loop:
                break
        if end:
            break
    model.save(os.path.basename(sys.argv[0]) + '_' + str(task) + '_final.h5')
