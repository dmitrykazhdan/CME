import os
import tensorflow as tf
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization


def build_dsprites_model(input_shape, num_classes=10):

    model = tf.keras.Sequential([

        Conv2D(64, (8, 8), strides=(2, 2), padding='same', input_shape=input_shape),
        Dropout(0.3),

        Conv2D(128, (6, 6), strides=(2, 2), padding='valid'),
        BatchNormalization(),

        Conv2D(128, (5, 5), strides=(1, 1), padding='valid'),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.Adam(lr=1e-4, amsgrad=True)

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model



def get_dsprites_model(train_x, train_y, models_dir, task="", num_classes=10,
                       checkpoint=True,
                       batch_size=128,
                       epochs=250):

    input_shape = train_x.shape[1:]
    model = build_dsprites_model(input_shape, num_classes)

    model_name = "dsprites_model"
    model_name = model_name + "_{}".format(task) if task != "" else model_name

    model_save_path = models_dir + model_name

    if not os.path.exists(model_save_path):
        # Train model
        callbacks = []
        if checkpoint:
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_save_path, verbose=True,
                                                             save_best_only=True, monitor='val_loss', mode='min', save_freq=50)
            callbacks.append(cp_callback)

        model.fit(x=train_x, y=train_y, epochs=epochs, batch_size=batch_size, validation_split=0.15,
                  callbacks=callbacks)

        model.save_weights(model_save_path)

    else:
        model.load_weights(model_save_path)

    return model

