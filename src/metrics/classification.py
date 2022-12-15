import tensorflow as tf
from tensorflow.keras import layers, Sequential, initializers

from sklearn.linear_model import LogisticRegression

from src.data.abstract_dataset import DatasetType


def make_cnn_model(input_shape, n_classes=10):
    # Copied from "Don't Generate Me: Training Differentially Private Generative Models with Sinkhorn Divergence", Cao et al.
    model = Sequential()
    model.add(layers.Conv2D(32, (3, 3), strides=1, input_shape=input_shape))
    model.add(layers.MaxPool2D(2))
    model.add(layers.ReLU())
    model.add(layers.Dropout(0.5))

    model.add(layers.Conv2D(64, (3, 3), strides=1))
    model.add(layers.MaxPool2D(2))
    model.add(layers.ReLU())
    model.add(layers.Dropout(0.5))

    model.add(layers.Conv2D(128, (3, 3), strides=1, activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(n_classes, activation='softmax'))

    from tensorflow_addons.optimizers import AdamW

    #my_adam = extend_with_decoupled_weight_decay(tf.keras.optimizers.Adam)
    optimizer = AdamW(weight_decay=1e-4, learning_rate=1e-3)#my_adam(weight_decay=1e-6, learning_rate=1e-4)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def make_cnn_model2(input_shape, n_classes=10):
    # Copied from "DP^^VAE" (Hiang et al.)
    model = Sequential()
    model.add(layers.Conv2D(32, (3, 3), strides=2, input_shape=input_shape))
    model.add(layers.Dropout(0.5))
    model.add(layers.ReLU())

    model.add(layers.Conv2D(64, (3, 3), strides=2))
    model.add(layers.Dropout(0.5))
    model.add(layers.ReLU())

    model.add(layers.Flatten())
    model.add(layers.Dense(n_classes, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def make_mlp_model(n_classes=10):
    # Copied from "Don't Generate Me: Training Differentially Private Generative Models with Sinkhorn Divergence", Cao et al.
    # Used also in "DP^^VAE" (Hiang et al.)
    model = Sequential()
    model.add(layers.Flatten(input_shape=(28, 28, 1)))
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dense(n_classes, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def test_LR_model(X_train, y_train, X_test, y_test, verbose=0):
    lr_model = LogisticRegression(solver='lbfgs', max_iter=5000, verbose=verbose)

    lr_model.fit(X_train, y_train)
    return lr_model.score(X_test, y_test)


def make_cifar100_simplenet_model(n_classes=20):
    model = Sequential()

    model.add(layers.Conv2D(64, (3, 3), strides=1, padding='same', kernel_initializer=initializers.glorot_uniform(), input_shape=(32, 32, 3)))
    model.add(layers.BatchNormalization(epsilon=1e-5, momentum=0.95))
    model.add(layers.ReLU())
    for i in range(3):
        model.add(layers.Conv2D(128, (3, 3), strides=1, padding='same', kernel_initializer=initializers.glorot_uniform()))
        model.add(layers.BatchNormalization(epsilon=1e-5, momentum=0.95))
        model.add(layers.ReLU())
    model.add(layers.MaxPool2D((2, 2), strides=2))
    model.add(layers.Dropout(0.1))
    for chs in [128, 128, 256]:
        model.add(layers.Conv2D(chs, (3, 3), strides=1, padding='same', kernel_initializer=initializers.glorot_uniform()))
        model.add(layers.BatchNormalization(epsilon=1e-5, momentum=0.95))
        model.add(layers.ReLU())
    model.add(layers.MaxPool2D((2, 2), strides=2))
    model.add(layers.Dropout(0.1))
    for i in range(2):
        model.add(layers.Conv2D(256, (3, 3), strides=1, padding='same', kernel_initializer=initializers.glorot_uniform()))
        model.add(layers.BatchNormalization(epsilon=1e-5, momentum=0.95))
        model.add(layers.ReLU())
    model.add(layers.MaxPool2D((2, 2), strides=2))
    model.add(layers.Dropout(0.1))
    model.add(layers.Conv2D(512, (3, 3), strides=1, padding='same', kernel_initializer=initializers.glorot_uniform()))
    model.add(layers.BatchNormalization(epsilon=1e-5, momentum=0.95))
    model.add(layers.ReLU())
    model.add(layers.MaxPool2D((2, 2), strides=2))
    model.add(layers.Dropout(0.1))
    for chs in [2048, 256]:
        model.add(layers.Conv2D(chs, (1, 1), strides=1, padding='valid', kernel_initializer=initializers.glorot_uniform()))
        model.add(layers.BatchNormalization(epsilon=1e-5, momentum=0.95))
        model.add(layers.ReLU())
    model.add(layers.MaxPool2D((2, 2), strides=2))
    model.add(layers.Dropout(0.1))
    model.add(layers.Conv2D(256, (3, 3), strides=1, padding='same', kernel_initializer=initializers.glorot_uniform()))
    model.add(layers.BatchNormalization(epsilon=1e-5, momentum=0.95))
    model.add(layers.ReLU())
    model.add(layers.GlobalMaxPool2D())
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(n_classes, activation='softmax'))

    # optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, decay=0.005, nesterov=False)
    # optimizer=tf.keras.optimizers.Adadelta(learning_rate=0.1, rho=0.9, epsilon=1e-3, decay=0.001)
    model.compile(optimizer=tf.keras.optimizers.Adadelta(learning_rate=0.1, rho=0.9, epsilon=1e-3, decay=0.001),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


def make_cxr_classifier_model(data_dim):
    model = Sequential()
    model.add(layers.Conv2D(32, (3, 3), strides=1, padding='same', activation='relu', input_shape=(data_dim, data_dim, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D((2, 2), strides=2, padding='same'))
    model.add(layers.Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(layers.Dropout(0.1))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D((2, 2), strides=2, padding='same'))
    model.add(layers.Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D((2, 2), strides=2, padding='same'))
    model.add(layers.Conv2D(128, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D((2, 2), strides=2, padding='same'))
    model.add(layers.Conv2D(256, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D((2, 2), strides=2, padding='same'))
    model.add(layers.Flatten())
    model.add(layers.Dense(units=128, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(units=1, activation='sigmoid'))
    model.compile(optimizer="rmsprop", loss='binary_crossentropy', metrics=['accuracy', 'AUC'])

    return model


def make_densenet121_model(data_dim, n_classes, pretrained=True):
    if pretrained:
        weights = 'imagenet'
        input_shape = (data_dim, data_dim, 3)
    else:
        weights = None
        input_shape = (data_dim, data_dim, 1)
    densenet = tf.keras.applications.densenet.DenseNet121(include_top=False, weights=weights,
                                                          input_shape=input_shape,
                                                          pooling='avg')
    image_input = layers.Input(shape=(data_dim, data_dim, 1))

    if pretrained:
        densenet.trainable = False
        preprocessed_input = layers.Concatenate()([image_input, image_input, image_input])
        preprocessed_input = preprocessed_input * 255.0
        preprocessed_input = tf.keras.applications.densenet.preprocess_input(preprocessed_input)
        output = densenet(preprocessed_input, training=False)
    else:
        preprocessed_input = image_input * 2 - 1
        output = densenet(preprocessed_input)

    output = tf.keras.layers.Dense(512, activation="relu")(output)
    output = tf.keras.layers.Dropout(0.25)(output)
    output = tf.keras.layers.Dense(n_classes, activation="softmax")(output)

    model = tf.keras.Model(inputs=[image_input], outputs=[output])
    optimizer = tf.keras.optimizers.Adam(0.001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def make_classifier_model(data_config):
    dataset_type = DatasetType.from_value(data_config.name)
    input_shape = (data_config.data_dim, data_config.data_dim, data_config.data_ch)

    if dataset_type == DatasetType.MNIST:
        return make_cnn_model(input_shape, data_config.n_classes)
    elif dataset_type == DatasetType.FEMNIST:
        return make_cnn_model(input_shape, data_config.n_classes)
    elif dataset_type == DatasetType.FMNIST:
        return make_cnn_model(input_shape, data_config.n_classes)
    elif dataset_type == DatasetType.FASHION_MNIST:
        return make_cnn_model(input_shape, data_config.n_classes)
    elif dataset_type == DatasetType.CELEBA:
        return make_cnn_model(input_shape, data_config.n_classes)
    elif dataset_type == DatasetType.CIFAR100:
        return make_cifar100_simplenet_model(n_classes=data_config.n_classes)
    elif dataset_type in [DatasetType.CXR]:
        return make_cxr_classifier_model(data_config.data_dim)
    elif dataset_type == DatasetType.CHEXPERT:
        return make_densenet121_model(data_config.data_dim, data_config.n_classes, pretrained=False)
    else:
        raise ValueError(f"Dataset {dataset_type.value} not supported.")


def run_classifier(model, train_dataset, test_dataset, val_dataset=None, epochs=10, verbose=1):
    early_stopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, verbose=verbose, restore_best_weights=True)
    if val_dataset is None:
        train_hist = model.fit(train_dataset, validation_split=0.1, epochs=epochs, verbose=verbose, shuffle=False)
    else:
        train_hist = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs, verbose=verbose, callbacks=[early_stopper])
    test_hist = model.evaluate(test_dataset, verbose=verbose)

    if verbose == 1:
        from sklearn.metrics import confusion_matrix
        import numpy as np

        y_labels = []
        y_true = []
        for batch in test_dataset.as_numpy_iterator():
            images, labels = batch
            batch_y_pred = model.predict(images)
            batch_y_labels = np.argmax(batch_y_pred, axis=1)
            y_labels += batch_y_labels.tolist()
            y_true += labels.tolist()

        cm = confusion_matrix(y_true=y_true, y_pred=y_labels)
        print(cm)

    return test_hist[1], train_hist
