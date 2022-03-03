import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
from numpy.random import seed
seed(333)
import tensorflow as tf
tf.random.set_seed(333)


def get_model(input_shape):
    from tensorflow.keras import Sequential
    from tensorflow.keras import Input
    from tensorflow.keras.layers import InputLayer, Conv2D, MaxPooling2D, Flatten, Dot, Lambda, Dense, Dropout, BatchNormalization, AveragePooling2D, UpSampling2D, Conv2DTranspose, Activation
    import tensorflow as tf
    import tensorflow.keras.backend as K
    from tensorflow.keras import activations

    # Define the tensors for the two input images
    
    act="relu"
    #act="sigmoid"
    # Convolutional Neural Network
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv2D(16, (3, 3), activation=act))
    model.add(Dropout(0.7))
    model.add(AveragePooling2D())

    model.add(Conv2D(16, (3, 3), activation=act))
    model.add(Dropout(0.7))
    model.add(AveragePooling2D())
    
    model.add(Conv2D(32, (3, 3), activation=act))
    model.add(Dropout(0.7))
    model.add(AveragePooling2D())
    
    model.add(Conv2D(32, (3, 3), activation=act))
    model.add(Dropout(0.7))
    model.add(AveragePooling2D())
    
    model.add(Conv2D(64, (3, 3), activation=act))
    model.add(Dropout(0.7))

    model.add(Flatten())
   
    model.add(Dropout(0.7))
    model.add(Dense(2048, activation=act))

    
    
    model.summary()
   
    return model

def contrastive_acc(model, loss_tmp_t, loss_tmp_v):
    import tensorflow as tf
    import tensorflow.keras.backend as K
    from tensorflow.python.ops import math_ops
    loss_tmp=loss_tmp_t
    
    

    #print("loss_tmp: {}".format(np.shape(loss_tmp)))
    if K.learning_phase==1:
        y_ref = model.predict(np.array(loss_tmp_t))
    else:
        y_ref = model.predict(np.array(loss_tmp_v))
    y_ref2=tf.constant(np.asarray(y_ref, np.float32), tf.float32)

   
    
    def acc(y_true, y_pred):
       
        margin = 0.5
        diff = K.abs(y_pred - y_ref2)
        margin = math_ops.cast(margin, y_pred.dtype)
        
        diff = math_ops.cast(diff < margin, y_pred.dtype)
       
        return K.mean(math_ops.equal(diff, 1), axis=-1)
        #return 1-K.mean(((y_true)*(1/2)*(K.square(K.maximum(margin - diff, 0)))) + ((1-y_true)*(1/2)*(K.square(diff))))
    return acc



  
  


def contrastive_loss(model, loss_tmp_t, loss_tmp_v):
    import tensorflow as tf
    
    import tensorflow.keras.backend as K
    from tensorflow.keras.layers import Lambda
    import numpy as np
    
    
    if K.learning_phase() == 1:
        res = model.predict(np.array(loss_tmp_t))
            
    else:
        res = model.predict(np.array(loss_tmp_v))
    y_ref=tf.constant(np.asarray(res), dtype=np.float32)
    
    def loss(y_true, y_pred):  
        margin = 1

        diff = K.abs(y_pred - y_ref)
        
        return K.mean(((y_true)*(1/2)*K.square(margin - diff)) + ((1-y_true)*(1/2)*K.square(diff)))
       
  
    return loss


if __name__ == '__main__':
    
    import numpy as np
   
    import sys
    
    from tensorflow.keras import Model
    import platform
    
    from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau
    from tensorflow.keras import optimizers
    
    
    from tensorflow.keras.preprocessing import image
    import gc, random
    
    
    if (platform.system() == 'Linux'):
        work_dir = 'saved_models/'
        #dataset_path = "/home/azam/task3/GardensPointWalking/"
        dataset_path = "/home/azam/task3/places4/"
        #dataset_path = "dataset/"              

    else:
        if platform.node() == 'DESKTOP-3R4O091':
            work_dir = 'H:\\Research_heavyData\\saved_models\\task3\\'
            #dataset_path = "H:/Research_heavyData/dogs-vs-cats/train/"       # Laptop
            #dataset_path = "F:/Research_heavyData/Datasets/GardensPointWalking/"       # Laptop
            dataset_path = "H:/Research_heavyData/siamese/places4/"       # Laptop
            #testset_path = "H:/Research_heavyData/dogs-vs-cats/train/"       # Laptop
            
            #dataset_path = "F:/Research_heavyData/Datasets/GardensPointWalking/combined/"       # Laptop
            #dataset_path = "F:/Research_heavyData/Datasets/CampusLoopDataset/CampusLoopDataset/combined/"
            #dataset_path = "F:/Research_heavyData/Datasets/angle/obj10/"       # Laptop
            
        else:
            work_dir = 'C:\\tmp\\'
            #dataset_path = "F:/research_datasets/GardensPointWalking/Images_day_left_day_right/"
            #dataset_path = "F:/research_datasets/CampusLoopDataset/CampusLoopDataset/images/"
            #dataset_path = "F:/research_datasets/SouthBankBicycle/images/"
   
    categories_traning = [os.path.join(dataset_path + "training_set", o) for o in os.listdir(dataset_path + "training_set") 
                    if os.path.isdir(os.path.join(dataset_path + "training_set",o))]
    categories_traning.sort()
    
    
    
    
    files_t=[]
    labels_t=[]
    
    for cat in categories_traning:
        tmp=os.listdir(os.path.join(dataset_path+ "training_set", cat))
        files_t.extend(tmp)
        labels_t.extend(os.path.basename(cat) for d in tmp)
        
    # Validation Data generation
    categories_validation = [os.path.join(dataset_path + "validation_set", o) for o in os.listdir(dataset_path + "validation_set") 
                    if os.path.isdir(os.path.join(dataset_path + "validation_set",o))]
    categories_validation.sort()
    
    files_v=[]
    labels_v=[]
    
    for cat in categories_validation:
        tmp=os.listdir(os.path.join(dataset_path+ "validation_set", cat))
        files_v.extend(tmp)
        labels_v.extend(os.path.basename(cat) for d in tmp)
    
    input_dims=(100,100)

    print("number of categories in training set: {} and in validation set: {}".format(len(categories_traning), len(categories_validation)))
    print("number of files in training set: {} and in validation set: {}".format(len(files_t), len(files_v)))
    model = get_model((input_dims[0], input_dims[1], 1))
   

    
    
    
    chk = ModelCheckpoint(work_dir + sys.argv[0] + '.ep={epoch:02d},loss={loss:.2f},acc={acc:.2f},vl={val_loss:.2f},va={val_acc:.2f}.h5', save_weights_only=True, monitor='val_loss', save_best_only=True, period=9) 
    csv_logger = CSVLogger(work_dir + sys.argv[0] + 'training.log')
    #rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.99, patience=49, verbose=1, mode='auto', min_delta=0.00001, cooldown=10, min_lr=0.0001)
    # callbacks_list = [chk, csv_logger, TestCallback((testset), save_data=True, save_path=work_dir + sys.argv[0] + 'test_data.log')]
    #tb = tf.keras.callbacks.TensorBoard(log_dir='./logs')
    #cb = [chk, csv_logger, rlr]
    cb = [csv_logger]
    
    
    
    y_train=[]
    z_train=[]
    loss_tmp_t=[]
    loss_tmp_v=[]
   
    count=0
    loss_th=0.3
    vloss_th = 0.3
    acc_th = 0.8
    rnd=0
    i=0
    v=0
    x_val=[]
    y_val=[]
    res_val=[]
    
    zeros=np.zeros((128))
    #match=[]
    img_dr = np.array(image.load_img(os.path.join(dataset_path + "training_set/" + labels_t[0], files_t[0]), target_size=input_dims, grayscale=True))/255
    img_dr = np.expand_dims(img_dr, 2)
    loss_tmp_t.append(img_dr)
    loss_tmp_v.append(img_dr)
    
    #match.append(1)
    #opt = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-07)   # 0.1850 loss min
    #opt = optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True, name="SGD")
    #opt = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name="Adam")
    #opt = optimizers.Adagrad(lr=0.01, initial_accumulator_value=0.1, epsilon=1e-07, name="Adagrad")
    opt = optimizers.Adadelta(lr=0.01, rho=0.33, epsilon=1e-07, name="Adadelta")

   
    model.compile(optimizer =opt, loss = contrastive_loss(model, loss_tmp_t, loss_tmp_v), metrics = [contrastive_acc(model, loss_tmp_t, loss_tmp_v)])
    loss_tmp_t=[]
    loss_tmp_v=[]
    
    epochs=50
    
   
    for i in range(0, len(files_t)-1):
        print("i={}".format(i))
        last_value_matched = False
        img_i = np.array(image.load_img(os.path.join(dataset_path + "training_set/" + labels_t[i], files_t[i]), target_size=input_dims, grayscale=True))/255
        img_i = np.expand_dims(img_i, 2)
        y_train.append(img_i)       # True sample
        y_train.append(img_i)       # True sample
        
        # another data for comparison
        # true training sample - output should be minimum
        img_j = np.array(image.load_img(os.path.join(dataset_path + "training_set/" + labels_t[i+1], files_t[i+1]), target_size=input_dims, grayscale=True))/255
        #y_ref_t = model.predict(np.array(np.expand_dims(img_j, 0)))
        img_j = np.expand_dims(img_j, 2)
        loss_tmp_t.append(img_j)
        
        if labels_t[i] == labels_t[i+1]:
            z_train.append(0.)
            last_value_matched = True
        else:
            z_train.append(1.)
            last_value_matched = False
            
        #print("true match i={}, labels[i]={}, labels[i+1]={}, file[i]={}, file[i+1]={}".format(i, labels_t[i], labels_t[i+1], files_t[i], files_t[i+1]))
        
        if last_value_matched:
            # getting false match
            diff_cat = dataset_path + "training_set/" + labels_t[i]
            
            while(diff_cat == (dataset_path + "training_set/" + labels_t[i])):
                diff_cat = dataset_path + "training_set/" + labels_t[random.randint(0, len(labels_t)-1)]
            print("current cat: {}, diff cat: {}".format(labels_t[i], diff_cat))
            files_in_diff_cat = os.listdir(diff_cat)
            #print("files in diff cat: {}".format(files_in_diff_cat))
            
            rnd=random.randint(0, len(files_in_diff_cat)-1)
            img_j = np.array(image.load_img(os.path.join(diff_cat, files_in_diff_cat[rnd]), target_size=input_dims, grayscale=True))/255
            img_j = np.expand_dims(img_j, 2)
            loss_tmp_t.append(img_j)
            if dataset_path + "training_set/" + labels_t[i] == diff_cat:
                z_train.append(0.)
            else:
                z_train.append(1.)
            #print("2nd false match loss i={}, labels[i]={}, labels[rnd]={}, file[i]={}, file[rnd]={}, y_true2: {}".format(i, labels_t[i], diff_cat, files_t[i], files_in_diff_cat[rnd], z_train[-1]))
        else:
            # getting true match
            files_in_same_cat = os.listdir(dataset_path + "training_set/" + labels_t[i])
            #print("files in same cat: {}".format(files_in_same_cat))
            #print("files_t[i+1]: {}".format(files_t[i]))
            #print("files_in_same_cat[0]: {}".format(files_in_same_cat[0]))
            rnd=random.randint(0, len(files_in_same_cat)-1)
            while (files_in_same_cat[rnd]==files_t[i]):
                rnd=random.randint(0, len(files_in_same_cat)-1)
            img_j = np.array(image.load_img(os.path.join(dataset_path + "training_set/" + labels_t[i], files_in_same_cat[rnd]), target_size=input_dims, grayscale=True))/255
            #y_ref_t2 = model.predict(np.array(np.expand_dims(img_j, 0)))
            loss_tmp_t.append(img_j)
            if labels_t[i] == labels_t[i]:
                z_train.append(0.)
            else:
                z_train.append(1.)
            #print("2nd true match loss i={}, labels[i]={}, labels[rnd]={}, file[i]={}, file[rnd]={}, z_train: {}".format(i, labels_t[i], labels_t[i], files_t[i], files_in_same_cat[rnd], z_train[-1]))
        
        # ===================================================================================================
        # Vaidation data for comparison
        last_value_matched = False
        print("validation phase executing")

        if (v > (len(files_v)-1)):
            v=0

        # validation file from same category
        img_i = np.array(image.load_img(os.path.join(dataset_path + "validation_set/" + labels_v[v], files_v[v]), target_size=input_dims, grayscale=True))/255
        img_i = np.expand_dims(img_i, 2)
        x_val.append(img_i)
        x_val.append(img_i)




        if (v+1)> (len(files_v)-1):
            v=-1
        #else:
            #v=m
        img_j = np.array(image.load_img(os.path.join(dataset_path + "validation_set/" + labels_v[v+1], files_v[v+1]), target_size=input_dims, grayscale=True))/255
        #y_ref_t = model.predict(np.array(np.expand_dims(img_j, 0)))
        img_j = np.expand_dims(img_j, 2)

        loss_tmp_v.append(img_j)
        if labels_v[v] == labels_v[v+1]:
            y_val.append(0.)
            last_value_matched = True
        else:
            y_val.append(1.)
            last_value_matched = False
            
        #print("true validation match v={}, labels[v]={}, labels[v+1]={}, file[v]={}, file[v+1]={}".format(v, labels_v[v], labels_v[v+1], files_v[v], files_v[v+1]))
        
        if last_value_matched:
            # getting false match
            diff_cat = dataset_path + "validation_set/" + labels_v[v]
            
            while(diff_cat == (dataset_path + "validation_set/" + labels_v[v])):
                diff_cat = dataset_path + "validation_set/" + labels_v[random.randint(0, len(labels_v)-1)]
            #print("current cat: {}, diff cat: {}".format(labels_v[v], diff_cat))
            files_in_diff_cat = os.listdir(diff_cat)
#            print("files in diff cat: {}".format(files_in_diff_cat))
            
            rnd=random.randint(0, len(files_in_diff_cat)-1)
            img_j = np.array(image.load_img(os.path.join(diff_cat, files_in_diff_cat[rnd]), target_size=input_dims, grayscale=True))/255
            #y_ref_t2 = model.predict(np.array(np.expand_dims(img_j, 0)))
            img_j = np.expand_dims(img_j, 2)
            loss_tmp_v.append(img_j)
            if dataset_path + "validation_set/" + labels_v[v] == diff_cat:
                y_val.append(0.)
            else:
                y_val.append(1.)
            #print("2nd false validation match loss v={}, labels[v]={}, labels[rnd]={}, file[v]={}, file[rnd]={}, y_val: {}".format(v, labels_v[v], diff_cat, files_v[v], files_in_diff_cat[rnd], y_val[-1]))
        else:
            # getting true match
            files_in_same_cat = os.listdir(dataset_path + "validation_set/" + labels_v[v])
            #print("files in same cat: {}".format(files_in_same_cat))
            #print("files_v[v]: {}".format(files_v[v]))
            #print("files_in_same_cat[0]: {}".format(files_in_same_cat[0]))
            rnd=random.randint(0, len(files_in_same_cat)-1)
            while (files_in_same_cat[rnd]==files_v[v]):
                rnd=random.randint(0, len(files_in_same_cat)-1)
            img_j = np.array(image.load_img(os.path.join(dataset_path + "validation_set/" + labels_v[v], files_in_same_cat[rnd]), target_size=input_dims, grayscale=True))/255
            #y_ref_t2 = model.predict(np.array(np.expand_dims(img_j, 0)))
            img_j = np.expand_dims(img_j, 2)
            loss_tmp_v.append(img_j)
            if labels_v[v] == labels_v[v]:
                y_val.append(0.)
            else:
                y_val.append(1.)
            #print("2nd validation true match loss v={}, labels[v]={}, labels[rnd]={}, file[v]={}, file[rnd]={}, y_val: {}".format(v, labels_v[v], labels_v[v], files_v[v], files_in_same_cat[rnd], y_val[-1]))

        
        
    
       
        v=v+1
        if v > (len(files_v)-1):
            v=0
             
             
       
        del img_i
        gc.collect()
    
        loss=1
        vloss=1
        acc=0

        
        prev_loss = loss
        prev_acc = acc
        count=0
       
        if(i>5):
           epochs=20
        if (i % 5==0):
            model.save_weights(work_dir + sys.argv[0] + "_" + str(i) +"_model_weights.h5")

        while((loss > loss_th) or (vloss > vloss_th) or (acc < acc_th)):           # for embeddings under control
            print("Training on {} samples out of {} and validating on {} samples out of {}".format(len(y_train)/2, len(files_t), len(x_val)/2, len(files_v)))
            
            h=model.fit(np.array(y_train), np.array(z_train), batch_size=128, epochs=epochs, verbose=1, validation_data=(np.array(x_val), np.array(y_val)), callbacks=cb, shuffle=False, steps_per_epoch=None, use_multiprocessing=True, validation_steps=None)
            loss=h.history['loss'][-1]
            vloss=h.history['val_loss'][-1]
            acc=h.history['acc'][-1]

            count = count + 1
            if (not(loss==prev_loss)):
                prev_loss = loss
                prev_acc = acc
                count=0
            else:
                if count >= 15:
                    break
            
        
            
        
        gc.collect()
    cb = [csv_logger, chk]

    h=model.fit(np.array(y_train), np.array(z_train), batch_size=128, epochs=200, verbose=1, validation_data=(np.array(x_val), np.array(y_val)), callbacks=cb)
    model.save_weights(work_dir + sys.argv[0] + "_EUC_model_weights.h5")
    #model.save(work_dir + sys.argv[0] + "_model.h5")
    print("Saved model weights to disk")


# model = load_model('model.h5', custom_objects={'loss': asymmetric_loss(alpha)})

