
# Empaquetemos estos resultados en una funcion :
def load_dataset():
    (trainX, trainy), (testX, testy) = mnist.load_data()
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))
    trainy = to_categorical(trainy)
    testy = to_categorical(testy)
    return trainX, trainy, testX, testy



# Empaquetemos este procesamiento en una funcion 
def prep_pixels(train, test):
    """
    Preparamos los pixeles para ser procesados por operaciones 
    con  elementos flotantes y normalizamos 
    """
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    train_norm = train_norm/255
    test_norm = test_norm/255
    return train_norm, test_norm 





# Definimos una funcion para empaquetar nuestra arquitectura 
def define_model():
    model = Sequential()
    # Primera capa oculta : Especificar los input
    model.add(Conv2D(32, (3,3), 
                     activation='relu',
                     kernel_initializer= 'he_uniform',
                     input_shape = (28,28,1)))
    model.add(MaxPooling2D((2,2)))
    
    # Podemos agregar mas : Conv2D + MxPooling
    
    model.add(Flatten())
    model.add(Dense(100,activation = "relu", 
                    kernel_initializer= 'he_uniform'))
    model.add(Dense(10, activation='softmax') )
    opt = gradient_descent_v2.SGD(learning_rate= 0.01,
                                  momentum= 0.9)
    model.compile(optimizer= opt,
                  loss = "categorical_crossentropy",
                  metrics = 'accuracy')
    return model 


# EN vista de la naturaleza de los datos de entrada
# obliguemos al modelo a mejorarse en tiempo de entrenamiento (KFold)

# Evaluamos el modelo usando UNa validacion cruzada 
def evaluate_model(dataX, dataY , n_folds = 2):
    # guardamos los scores 
    scores = []
    # MOdelos ajustados : historia de mi modelo 
    histories = []
    # 
    # Instanciamos KFold
    kfold = KFold(n_folds, shuffle=True, random_state = 1)
    # 
    # Brrido sobre cada fold/pliegue de nuestro de datos de train
    for train_ix, test_ix in kfold.split(dataX):
        # Creamos el modelo 
        model = define_model()
        # Sleccionamos las fils para train y test 
        trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
        # Ajusto el modelo 
        history = model.fit(trainX, trainY,
                            epochs = 2,
                            batch_size = 32,
                            validation_data=(testX, testY),
                            verbose = 1)
        # Evaluamos el modelo 
        _, acc = model.evaluate(testX, testY)
        print("=> %.3f" %(acc*100))
        # Almacenamos los scores y histories
        scores.append(acc)
        histories.append(history)
    return scores, histories


# %% Probemos nuestras funciones
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# optimizador 
from keras.optimizers import gradient_descent_v2
 
from keras.datasets import mnist
# help(mnist.load_data)
(trainX, trainy), (testX, testy) = mnist.load_data()

# Cargar los datos 
trainX, trainY, testX, testY = load_dataset()

# preparamos los pixeles 
trainX , testX = prep_pixels(trainX, testX)

# COnstruimos el modelo y lo evaluamos 
sc1 , hst1 = evaluate_model(trainX, trainY)