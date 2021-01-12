import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional
from tensorflow.keras.models import Sequential
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report



def lstmf(x_train, y_train,x_test, y_test,actions, ler, cl,drp,ep):


    model = Sequential()
    model.add(Bidirectional(LSTM(cl, input_shape=x_train.shape[1:], activation='relu', return_sequences=True)))
    model.add(Dropout(drp))

    model.add(Bidirectional(LSTM(cl, activation='relu')))
    model.add(Dropout(drp))

    model.add(Dense(cl / 2, activation='relu'))
    model.add(Dropout(drp))

    model.add(Dense(len(actions)+1, activation='softmax'))

    opt = tf.keras.optimizers.Adam(lr=ler, decay=1e-6, clipnorm=1.0)

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                               patience=10,
                               verbose=True,
                               mode='min')

    model.fit(x_train, y_train, epochs=ep)
           #   callbacks = [early_stop])
    val_loss, val_acc = model.evaluate(x_test, y_test)



    pred = model.predict(x_test)

    pr = np.argmax(pred, axis=1)

    conf = confusion_matrix(y_test, pr)


    results = {"acc": val_acc, "conf": conf, "report": classification_report(y_test, pr), "predict": pr}

    return results