# https://www.tensorflow.org/tutorials/quickstart/beginner
import tensorflow as tf  # tf2.0
from termcolor import cprint


def load_dataset():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return (x_train, y_train), (x_test, y_test)


def init_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10),
    ])
    return model


if __name__ == '__main__':
    cprint('> Keras MNIST', 'green')
    (x_train, y_train), (x_test, y_test) = load_dataset()
    model = init_model()

    cprint('> smoke test', 'green')
    pred = model(x_train[:1]).numpy()
    spred = tf.nn.softmax(model(x_train[:1])).numpy()
    print(pred, y_train[:1])

    cprint('> loss function', 'green')
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True)
    print(loss_fn(y_train[:1], pred).numpy())

    cprint('> compile model', 'green')
    model.compile(optimizer='adam', loss=loss_fn,
                  metrics=['accuracy'])

    cprint('> training', 'green')
    model.fit(x_train, y_train, epochs=5)

    cprint('> evaluate', 'green')
    model.evaluate(x_test, y_test, verbose=2)

    cprint('> prob model', 'green')
    model_prob = tf.keras.Sequential([
        model, tf.keras.layers.Softmax()])
    print(model_prob(x_test[:5]), y_test[:5])
