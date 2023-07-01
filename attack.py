  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
tf.keras.utils.set_random_seed(0)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from scipy.special import softmax
from tqdm import tqdm
import argparse
import os
from torchvision.datasets import SVHN
import imp
import aggregation
import deep_cnn
import input  # pylint: disable=redefined-builtin
import numpy as np
from six.moves import xrange
from sklearn.metrics import classification_report, accuracy_score
from tensorflow import keras
from computation_utils import Target_Model_pred_fn
from computation_utils import fn_R_given_Selected
from computation_utils import fn_Sample_Generator
from computation_utils import fn_Jacobian_Calculation
from numpy import linalg as LA
import multiprocessing
import copy
from joblib import Parallel, delayed
from multiprocessing import Pool
from scipy.special import softmax
from scipy.stats import entropy
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.metrics import confusion_matrix
MODEL_PATH = './model7543/'
DATA_PATH = './data7543/'

def create_dir_if_needed(dest_directory):
  """Create directory if doesn't exist."""
  if not tf.gfile.IsDirectory(dest_directory):
    tf.gfile.MakeDirs(dest_directory)

  return True


if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

def maybe_download(file_urls, directory):
  """Download a set of files in temporary local folder."""

  # Create directory if doesn't exist
  assert create_dir_if_needed(directory)

  # This list will include all URLS of the local copy of downloaded files
  result = []

  # For each file of the dataset
  for file_url in file_urls:
    # Extract filename
    filename = file_url.split('/')[-1]

    # If downloading from GitHub, remove suffix ?raw=True from local filename
    if filename.endswith("?raw=true"):
      filename = filename[:-9]

    # Deduce local file url
    #filepath = os.path.join(directory, filename)
    filepath = directory + '/' + filename

    # Add to result list
    result.append(filepath)

    # Test if file already exists
    if not tf.gfile.Exists(filepath):
      def _progress(count, block_size, total_size):
        sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
            float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()
      filepath, _ = urllib.request.urlretrieve(file_url, filepath, _progress)
      print()
      statinfo = os.stat(filepath)
      print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

  return result

def load_trained_indices():
    fname = MODEL_PATH + 'data_indices.npz'
    with np.load(fname) as f:
        indices = [f['arr_%d' % i] for i in range(len(f.files))]
    return indices


def get_data_indices(data_size, target_train_size=int(1e4), sample_target_data=True):
    sample_target_data=True
    train_indices = np.arange(data_size)
    if sample_target_data:
        target_data_indices = np.random.choice(train_indices, target_train_size, replace=False)
        shadow_indices = np.setdiff1d(train_indices, target_data_indices)
        print("olaviyat target_data_indices len",len(target_data_indices))
        print("olaviyat shadow len",len(shadow_indices))
    else:
        target_data_indices = train_indices[:target_train_size]
        shadow_indices = train_indices[target_train_size:]
    return target_data_indices, shadow_indices


def load_attack_data():
    fname = MODEL_PATH + 'attack_train_data.npz'
    with np.load(fname) as f:
        train_x, train_y = [f['arr_%d' % i] for i in range(len(f.files))]
    fname = MODEL_PATH + 'attack_test_data.npz'
    with np.load(fname) as f:
        test_x, test_y = [f['arr_%d' % i] for i in range(len(f.files))]
    return train_x.astype('float32'), train_y.astype('int32'), test_x.astype('float32'), test_y.astype('int32')
def Average(lst):
    return sum(lst) / len(lst)

def ensemble_preds(dataset, nb_teachers, stdnt_data,train_y,newMethod):

  models=[]
  indices2=[]

  on_model=15
  indices3=[]
  np.random.seed(1000)
  for i in range(len(stdnt_data)):
    #tr_L_models[i, :-non_model] = teacherData[i, :]
    tmp = np.arange(nb_teachers)
    np.random.shuffle(tmp)
    indices3.append(tmp[:on_model])
  arr22=[]
  #print(indices3)
  for item in indices3:
    for item2 in item:
      arr22.append(item2)
  #print(arr22)
  x=np.array(arr22)
  unique, counts = np.unique(x, return_counts=True)
  print("marhale 1 indices3",indices3[0])
  print(np.asarray((unique, counts)).T)
  for teacher_id in xrange(nb_teachers):
    data2=[]
    data3=[]
    for i,item in enumerate(indices3):
      if teacher_id in item:
        if i==0:
          print("marhale 2 indices3",teacher_id)
        data2.append(stdnt_data[i])
        data3.append(train_y[i])

    
    model=deep_cnn.target_model_fn777()

    model.fit(
          np.array(data2),np.array(data3), epochs=35,use_multiprocessing=True
          #,callbacks=[callback]
      )

    models.append(model)
  newMethod=False
  if newMethod==False:
      train_y = np.array(train_y)
      bl=[]
      stdList=[]
      Pred=[]
     
      Pred10=[]
      batch_size=100
      len_t =  int(np.ceil(len(stdnt_data)/batch_size))
      for batch_ind in tqdm(range(len_t)):
        R_x2=[]
        local_samples2=[]
        end_idx = min(len(stdnt_data), (batch_ind+1)*batch_size)
        features = stdnt_data[batch_ind*batch_size: end_idx]
        #labels = test_labels[batch_ind*batch_size: end_idx]
        jacobi=False
        if jacobi==True:
          for ii, item2 in tqdm(enumerate(features)):
        
            R_x=item2.reshape(3072,)      
            R_x=R_x.reshape(1, -1)
            local_samples = fn_Sample_Generator(R_x, "null")
            local_samples=local_samples.reshape(len(local_samples),32,32,3)
            
            R_x=R_x.reshape(32,32,3)
            R_x2.append(R_x)
            for item4 in local_samples:
              local_samples2.append(item4)
          R_x2=np.array(R_x2)
        
          local_samples2=np.array(local_samples2)
          
          R_local_proba=np.zeros([nb_teachers,  R_x2.shape[0], 10])
          local_proba=np.zeros([nb_teachers, local_samples2.shape[0], 10])
          for teacher_id in range(nb_teachers):
            R_local_proba[teacher_id] = models[teacher_id](R_x2,training=False)
            #print("teacher",teacher_id)
            #print(models[teacher_id](R_x2,training=False))
            local_proba[teacher_id] = models[teacher_id](local_samples2,training=False)
        
        for iii, item3 in tqdm(enumerate(features)):
          #Jacobian_matrix = np.zeros([nb_teachers, 10, 3072])
          entropy_matrix = np.zeros([25, 1])
          #blacklist=[]
          #print(ii)
          #for teacher_id in xrange(nb_teachers):
            #for ite in indices3[teacher_id]:
          #  if teacher_id not in indices3[batch_ind*batch_size+iii]:
          #    if iii==0:
          #      print("marhale 3",teacher_id)
          #      print(indices3[batch_ind*batch_size+iii])
          #    blacklist.append(teacher_id) 
          for teacher_id in xrange(nb_teachers):
          #for teacher_id in range(nb_teachers): 
            #R_local_proba2 = R_local_proba[teacher_id][iii]
            #local_proba2 = local_proba[teacher_id][iii*3072:3072*(iii+1)]
            
            #Jacobian_matrix[teacher_id] = fn_Jacobian_Calculation(R_local_proba2, local_proba2, 3072, 10) 
            x20= copy.deepcopy(item3.reshape(1,32,32,3))
            entropy_matrix[teacher_id]=entropy(models[teacher_id](x20)[0])
            #entropy_matrix[teacher_id]=entropy(models[teacher_id](item3.reshape(1,32,32,3))[0])   
          #Jacobian_norms = LA.norm(Jacobian_matrix, axis=(1, 2))
          Allindexes=np.argsort(entropy_matrix[:, 0])
          #Allindexes=np.argsort(Jacobian_norms)

          Pred2=[]
          for id2 in Allindexes[20:25]:

            Pred2.append(models[id2](item3.reshape(1,32,32,3)))
          Pred10.append(np.array(Pred2).mean(axis=0))
      Pred10=np.array(Pred10)#lowest
      

      return Pred10.reshape(len(Pred10),10)




import math

import math
def loss(y_true,y_pred):
  cce = tf.keras.losses.SparseCategoricalCrossentropy()
  return cce(y_true, y_pred).numpy()
def Entropy_attack(prob_train,prob_test,label_train,label_test):
  in1=np.ones(5000)
  out1=np.zeros(5000)
  tr_data=[]
  for k,i in enumerate(prob_train):

    tr_data.append(entropy(i))

  te_data=[]
  for k2,i2 in  enumerate(prob_test):
    te_data.append(entropy(i2))
  whole_data = np.concatenate((np.array(tr_data), np.array(te_data)), axis = 0)
  y_attack=np.concatenate((np.array(in1), np.array(out1)), axis = 0)
  fpr2, tpr2, threshold =metrics.roc_curve(y_attack, whole_data, pos_label=1)
  print("AUC Entropy attack attack",1-metrics.auc(fpr2, tpr2))
  FPR=[]
  TPR=[]
  return 0
def MEntropy_attack(prob_train,prob_test,label_train,label_test):
  in1=np.ones(5000)
  out1=np.zeros(5000)
  tr_data=[]
  for k,i in enumerate(prob_train):
    tr_data.append(_m_entr_comp(np.array([i]), np.array([label_train[k]]))[0])
  te_data=[]
  for k2,i2 in  enumerate(prob_test):

    te_data.append(_m_entr_comp(np.array([i2]), np.array([label_test[k2]]))[0])
  whole_data = np.concatenate((np.array(tr_data), np.array(te_data)), axis = 0)

  y_attack=np.concatenate((np.array(in1), np.array(out1)), axis = 0)
  fpr2, tpr2, threshold =metrics.roc_curve(y_attack, whole_data, pos_label=1)
  print("AUC modified entropy attack",1-metrics.auc(fpr2, tpr2))
  FPR=[]
  TPR=[]
  return 0
def Loss_attack(prob_train,prob_test,label_train,label_test):
  in1=np.ones(5000)
  out1=np.zeros(5000)
  tr_data=[]
  for k,i in enumerate(prob_train):
    tr_data.append(loss(np.array([label_train[k]]),np.array([i])))
  te_data=[]
  for k2,i2 in  enumerate(prob_test):

    te_data.append(loss( np.array([label_test[k2]]),np.array([i2])))

  whole_data = np.concatenate((np.array(tr_data), np.array(te_data)), axis = 0)
  y_attack=np.concatenate((np.array(in1), np.array(out1)), axis = 0)
  fpr2, tpr2, threshold =metrics.roc_curve(y_attack, whole_data, pos_label=1)
  print("AUC Loss attack",1-metrics.auc(fpr2, tpr2))

  return 0
def Gap_attack(prob_train,prob_test,label_train,label_test):
  in1=np.zeros(5000)
  in2=np.ones(5000)
  Attack_target=np.hstack((in2,in1))
  result1=np.argmax(prob_train,axis=1)
  result2=np.argmax(prob_test,axis=1)
  result_label=np.hstack((result1,result2)
  target_label=np.hstack((label_train,label_test))
  target_label=np.array(target_label)
  
  g1=len(target_label)
  target_label=target_label.reshape(g1,1)
  result_label=np.array(result_label)
  result_label=result_label.reshape(g1,1)
  target_label=target_label.astype(int)
  result_label=result_label.astype(int)
  attack_result=np.equal(target_label, result_label)
  attack_result=attack_result.astype(int) 
  fpr2, tpr2, thresholds = metrics.roc_curve(Attack_target, attack_result, pos_label=1)
  AUC=metrics.auc(fpr2, tpr2)
  print("AUC Gap Attack",AUC)
  return 0

def _log_value(probs, small_value=1e-30):
        return -np.log(np.maximum(probs, small_value))
def _m_entr_comp( probs, true_labels):
        log_probs = _log_value(probs)
        #print(log_probs)
        reverse_probs = 1-probs
        #print(reverse_probs)
        log_reverse_probs = _log_value(reverse_probs)

        modified_probs = np.copy(probs)
        modified_probs[range(true_labels.size), true_labels] = reverse_probs[range(true_labels.size), true_labels]
        modified_log_probs = np.copy(log_reverse_probs)
        modified_log_probs[range(true_labels.size), true_labels] = log_probs[range(true_labels.size), true_labels]
        return np.sum(np.multiply(modified_probs, modified_log_probs),axis=1)



def train_target_model(dataset, epochs=40 ,learning_rate=0.01):
    train_x, train_y, test_x, test_y = dataset

    teachers_preds = ensemble_preds('cifar10', 25, train_x,train_y,False)
    
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10, # rotation
        zoom_range=0.2, # zoom

    )

    model=deep_cnn.target_model_fn777V3()
    a=np.array(train_y)
    b = np.zeros((a.size, a.max() + 1))
    b[np.arange(a.size), a] = 1
    a2=np.array(test_y)
    b2 = np.zeros((a2.size, a2.max() + 1))
    b2[np.arange(a2.size), a2] = 1
    class MyCustomCallback(tf.keras.callbacks.Callback):
      def on_epoch_end(self, epoch, logs=None):
          res_eval_1 = self.model.evaluate(train_x, b, verbose = 0)
          res_eval_2 = self.model.evaluate(test_x, b2, verbose = 0)
          print("train, test",res_eval_1+res_eval_2)
    my_val_callback = MyCustomCallback()
    model=deep_cnn.target_model_fn777()

    from keras.callbacks import ModelCheckpoint
    mcp_save = ModelCheckpoint(os.getcwd()+'/SVHN/Models/'+'SVHN/E-1Jacobian/L10Jacobian8.npy'+'502.hdf5', save_best_only=True, monitor='val_accuracy', mode='max')
    model.fit(
    datagen.flow(train_x, teachers_preds),
        #train_x,teachers_preds,
    #     train_x, b,
           epochs=epochs
          ,callbacks=[my_val_callback,mcp_save]
      )
    
    model=keras.models.load_model(os.getcwd()+'/SVHN/Models/'+'SELENA'+'502.hdf5')
    attack_x=[]
    attack_y=[]

    
    student_preds1=model.predict(train_x) 
    attack_x.append(student_preds1)
    attack_y.append(np.ones(len(student_preds1)))
    pred_y2=[]
    true_y=[]
    mem=[]
    for k,item in enumerate(student_preds1):

      pred_y2.append(np.argmax(item))

    student_preds2=model.predict(test_x)
    attack_x.append(student_preds2)
    attack_y.append(np.zeros(len(student_preds2)))
    pred_y3=[]
    Nonmem=[]
    for k2,item2 in enumerate(student_preds2):

      pred_y3.append(np.argmax(item2))
      

    Entropy_attack(student_preds1,student_preds2,train_y,test_y)
    MEntropy_attack(student_preds1,student_preds2,train_y,test_y)
    Loss_attack(student_preds1,student_preds2,train_y,test_y)
    Gap_attack(student_preds1,student_preds2,train_y,test_y)

    attack_x = np.vstack(attack_x)

    attack_y = np.concatenate(attack_y)
    attack_x = attack_x.astype('float32')
    attack_y = attack_y.astype('int32')
    classes = np.concatenate([train_y, test_y])
    return attack_x, attack_y, classes





def train_shadow_models(epochs=40):

    
    attack_x, attack_y = [], []
    classes = []
    for i in range(n_shadow):
        data = load_data('shadow{}_data.npz'.format(i))
        train_x, train_y, test_x, test_y = data
        model=deep_cnn.target_model_fn777()

        model.fit(
              train_x, train_y, epochs=epochs, use_multiprocessing=True
          )

   

        

        attack_i_x, attack_i_y = [], []
        student_preds1=model.predict(train_x,verbose=0) 
        attack_i_x.append(student_preds1)
        attack_i_y.append(np.ones(len(student_preds1)))
        student_preds2=model.predict(test_x,verbose=0)
        attack_i_x.append(student_preds2)
        attack_i_y.append(np.zeros(len(student_preds2)))
        
        print ('Gather training data for attack model')
        
        attack_x += attack_i_x
        attack_y += attack_i_y
        classes.append(np.concatenate([train_y, test_y]))
    # train data for attack model
    attack_x = np.vstack(attack_x)
    attack_y = np.concatenate(attack_y)
    attack_x = attack_x.astype('float32')
    attack_y = attack_y.astype('int32')
    classes = np.concatenate(classes)
    
    return attack_x, attack_y, classes

   
def train_attack_model(classes, dataset=None, n_hidden=50, learning_rate=0.01, batch_size=200, epochs=50,
                       model='nn', l2_ratio=1e-7):
 
  
    if dataset is None:
        dataset = load_attack_data()

    train_x, train_y, test_x, test_y = dataset
    unique, counts = np.unique(train_y, return_counts=True)

    result = np.column_stack((unique, counts)) 
    unique, counts = np.unique(test_y, return_counts=True)

    result = np.column_stack((unique, counts)) 
    train_classes, test_classes = classes
    train_indices = np.arange(len(train_x))
    test_indices = np.arange(len(test_x))
    unique_classes = np.unique(train_classes)
    true_y = []
    pred_y = []
    pred_y2 = []
  
    from sklearn.metrics import precision_score
    for c in unique_classes:
        print ('Training attack model for class {}...'.format(c))
        c_train_indices = train_indices[train_classes == c]
        c_train_x, c_train_y = train_x[c_train_indices], train_y[c_train_indices]
        c_test_indices = test_indices[test_classes == c]
        c_test_x, c_test_y = test_x[c_test_indices], test_y[c_test_indices]
        
        
        c_dataset = (c_train_x, c_train_y, c_test_x, c_test_y)
        from sklearn.neural_network import MLPClassifier
        
        model=deep_cnn.attack_model_fn()
        from tensorflow.keras.callbacks import EarlyStopping
        early_stopping = EarlyStopping(patience=5,monitor='val_accuracy')
        model.fit(c_train_x, c_train_y, epochs=40,verbose=0,validation_data=(c_test_x, c_test_y)
       ,callbacks=[early_stopping]
        )
        da=model.predict(c_test_x,verbose=0)
    
        c_pred_y=[]
        for item in da:
          c_pred_y.append(np.argmax(item))
          
        c_pred_y2 = da[:, 1]  

        true_y.append(c_test_y)
        pred_y.append(c_pred_y)
        pred_y2.append(c_pred_y2)
    from sklearn.metrics import recall_score
    from matplotlib import pyplot as plt
    from sklearn import metrics
    print ('-' * 10 + 'FINAL EVALUATION' + '-' * 10 + '\n')
    true_y = np.concatenate(true_y)
    pred_y = np.concatenate(pred_y)
    pred_y2 = np.concatenate(pred_y2)
    print("PRECISION binary",precision_score(true_y, pred_y, average="binary"))
    print("recall binary",recall_score(true_y, pred_y, average="binary"))
    print ('Testing Accuracy: {}'.format(accuracy_score(true_y, pred_y)))
    print("roc_auc_score(y, clf.predict_proba(X)[:, 1])",roc_auc_score(true_y, pred_y2))
    fpr, tpr, thresholds = metrics.roc_curve(true_y,  pred_y2)
    print (classification_report(true_y, pred_y))


def save_data():
    print ('-' * 10 + 'SAVING DATA TO DISK' + '-' * 10 + '\n')

    import torch
    import torchvision
    dataset = SVHN(root='./data7', download=True)
    print(dataset.data.shape)
    print(dataset.labels.shape)
    #train_data=dataset.data[:5000]
    #train_labels=dataset.labels[:5000]
    #test_data=dataset.data[5000:10000]
    #test_labels=dataset.labels[5000:10000]
    X=dataset.data.astype(np.float32)
    #test_data=test_data.astype(np.float32)
    Y=dataset.labels.astype(np.int32)
    #train_data = train_labels.reshape(len(train_data),28,28,1)
    #test_labels=test_labels.astype(np.int32)
    #x = np.load(DATASET_FEATURES)
    #y = np.load(DATASET_LABELS)
    X = X.reshape(X.shape[0], 32, 32, 3)
    X = X / 255
    

    target_data_indices, shadow_indices = get_data_indices(len(X), target_train_size=args.target_data_size)
    np.savez(MODEL_PATH + 'data_indices.npz', target_data_indices, shadow_indices)
   
    # target model's data
    print ('Saving data for target model')
    train_x2, train_y2 = X[target_data_indices], Y[target_data_indices]
    size = len(target_data_indices)
    #train_x, test_x, train_y, test_y = train_test_split(x[target_data_indices], y[target_data_indices], test_size=0.5, random_state=42)
    #train_x, test_x, train_y, test_y = train_x2[:15000], train_x2[15000:], train_y2[:15000], train_y2[15000:]
    #test_x,test_y=test_x3[:5000], test_y3[:5000]
    #train_test_split(train_x2, train_y2,test_size=0.5)
    #test_x = test_x[target_data_indices]
    #test_y = test_y[:size]
    #if size < len(test_x):
    #    test_x = test_x[:size]
    #    test_y = test_y[:size]
    # save target data
    #print("train_x",len(train_x))
    np.savez(DATA_PATH + 'target_data.npz', train_x2[:5000], train_y2[:5000], train_x2[5000:], train_y2[5000:])
    
    # shadow model's data
    target_size = len(target_data_indices)
    shadow_x, shadow_y = X[shadow_indices], Y[shadow_indices]
    print("len shadow_x",len(shadow_x))
    shadow_indices = np.arange(len(shadow_indices))

    for i in range(args.n_shadow):
        print ('Saving data for shadow model {}'.format(i))
        #print(len(shadow_indices))
        #print(2 * target_size)
        shadow_i_indices = np.random.choice(shadow_indices, target_size, replace=False)
        #print("len(shadow)",len(shadow_i_indices))
        #print(np.ptp(shadow_i_indices))
        #print(np.min(shadow_i_indices)) 
        #print(np.max(shadow_i_indices))
        shadow_i_x, shadow_i_y = shadow_x[shadow_i_indices], shadow_y[shadow_i_indices]
        train_x, train_y = shadow_i_x[:5000], shadow_i_y[:5000]
        test_x, test_y = shadow_i_x[5000:], shadow_i_y[5000:]
        #test_x, test_y= test_x3[5000:], test_y3[5000:]
        #train_x, train_y = shadow_i_x, shadow_i_y
        #test_x, test_y= test_x3[5000:], test_y3[5000:]
        print("shadow data size train_x",len(train_x))
        print("shadow data size mehe test_x",len(test_x))
        np.savez(DATA_PATH + 'shadow{}_data.npz'.format(i), train_x, train_y, test_x, test_y)


def load_data(data_name):
    with np.load(DATA_PATH + data_name) as f:
        train_x, train_y, test_x, test_y = [f['arr_%d' % i] for i in range(len(f.files))]
    return train_x, train_y, test_x, test_y


def attack_experiment():
    
    dataset = load_data('target_data.npz')
      #attack_test_x, attack_test_y, test_classes = train_teacher(dataset, FLAGS.nb_teachers, FLAGS.teacher_id)
      attack_test_x, attack_test_y, test_classes = train_target_model(
          dataset=dataset,
          epochs=args.target_epochs,
          learning_rate=args.target_learning_rate,
          
          )
  
      #print ('-' * 10 + 'TRAIN SHADOW' + '-' * 10 + '\n')
      #print("tole attac test x",len(attack_test_x))
    attack_train_x, attack_train_y, train_classes = train_shadow_models(
          epochs=args.shadow_epochs,
          batch_size=args.shadow_batch_size,
          learning_rate=args.shadow_learning_rate,
          n_shadow=args.n_shadow,
          n_hidden=args.shadow_n_hidden,
          l2_ratio=args.shadow_l2_ratio,
          model=args.shadow_model,
          save=args.save_model)
  
      #print ('-' * 10 + 'TRAIN ATTACK' + '-' * 10 + '\n')
      #print("tole attac test x",len(attack_train_x))
    dataset = (attack_train_x, attack_train_y, attack_test_x, attack_test_y)
    train_attack_model(
          dataset=dataset,
          epochs=args.attack_epochs,
          batch_size=args.attack_batch_size,
          learning_rate=args.attack_learning_rate,
          n_hidden=args.attack_n_hidden,
          l2_ratio=args.attack_l2_ratio,
          model=args.attack_model,
          classes=(train_classes, test_classes))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('train_feat', type=str,default='train_feat')
    #parser.add_argument('train_label', type=str,default='train_label')
    parser.add_argument('--test_feat', type=str, default=None)
    parser.add_argument('--test_label', type=str, default=None)
    parser.add_argument('--save_model', type=int, default=1)
    parser.add_argument('--save_data', type=int, default=0)
    # if test not give, train test split configuration
    parser.add_argument('--test_ratio', type=float, default=0.3)
    # target and shadow model configuration
    parser.add_argument('--n_shadow', type=int, default=10)
    #parser.add_argument('--target_data_size', type=int, default=int(1e4))   # number of data point used in target model
    parser.add_argument('--target_data_size', type=int, default=10000)
    parser.add_argument('--target_model', type=str, default='cnn')
    parser.add_argument('--target_learning_rate', type=float, default=0.001)
    parser.add_argument('--target_batch_size', type=int, default=1)
    parser.add_argument('--target_n_hidden', type=int, default=256)
    parser.add_argument('--target_epochs', type=int, default=3)
    parser.add_argument('--target_l2_ratio', type=float, default=1e-15)
    
    parser.add_argument('--shadow_model', type=str, default='cnn')
    parser.add_argument('--shadow_learning_rate', type=float, default=0.001)
    parser.add_argument('--shadow_batch_size', type=int, default=1000)
    parser.add_argument('--shadow_n_hidden', type=int, default=256)
    parser.add_argument('--shadow_epochs', type=int, default=1)
    parser.add_argument('--shadow_l2_ratio', type=float, default=1e-6)

    # attack model configuration
    parser.add_argument('--attack_model', type=str, default='softmax')
    parser.add_argument('--attack_learning_rate', type=float, default=0.001)
    parser.add_argument('--attack_batch_size', type=int, default=1000)
    parser.add_argument('--attack_n_hidden', type=int, default=256)
    parser.add_argument('--attack_epochs', type=int, default=1)
    parser.add_argument('--attack_l2_ratio', type=float, default=1e-15)

    # parse configuration
    args = parser.parse_args()
    print (vars(args))
    if args.save_data:
        save_data()
    else:
        attack_experiment()
