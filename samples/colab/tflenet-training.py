
import numpy as np
import tensorflow as tf

BATCH_SIZE = 32
NUM_ROWS, NUM_COLS = 32, 32
EPOCHS = 20

import iree.compiler.tf
import iree.runtime
class Lenet5(tf.Module):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        
        inputs = tf.keras.layers.Input((NUM_ROWS, NUM_COLS, 1))
        x = tf.keras.layers.Conv2D(6, kernel_size=5)(inputs)
        x = tf.keras.layers.Activation("tanh")(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
        
        x = tf.keras.layers.Conv2D(16, kernel_size=5)(x)
        x = tf.keras.layers.Activation("tanh")(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
        
        x = tf.keras.layers.Conv2D(120, kernel_size=5)(x)
        x = tf.keras.layers.Activation("tanh")(x)
        x = tf.keras.layers.Flatten()(x)
        
        x = tf.keras.layers.Dense(84, activation='relu')(x)
        
        outputs = tf.keras.layers.Dense(units=self.output_shape, activation='softmax')(x)
        
        self.model = tf.keras.Model(inputs, outputs)
        
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=1e-2)
        
    @tf.function(input_signature=[
        tf.TensorSpec([BATCH_SIZE, NUM_ROWS, NUM_COLS, 1])  # inputs
    ])
    def predict(self, inputs):
        return self.model(inputs, training=False)

    # We compile the entire training step by making it a method on the model.
    @tf.function(input_signature=[
        tf.TensorSpec([BATCH_SIZE, NUM_ROWS, NUM_COLS, 1]),  # inputs
        tf.TensorSpec([BATCH_SIZE], tf.int32)  # labels
    ])
    def learn(self, inputs, labels):
        # Capture the gradients from forward prop...
        with tf.GradientTape() as tape:
            probs = self.model(inputs, training=True)
            loss = self.loss(labels, probs)
        # ...and use them to update the model's weights.
        variables = self.model.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return loss

if __name__ == '__main__':
    # Loading the dataset and perform splitting
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Resize image to 32x 32
    X_train = np.array([np.pad(X_train[i], pad_width=2) for i in range(X_train.shape[0])])
    X_test = np.array([np.pad(X_test[i], pad_width=2) for i in range(X_test.shape[0])])

    # Performing reshaping operation
    X_train = X_train.reshape(X_train.shape[0], NUM_ROWS, NUM_COLS, 1)
    X_test = X_test.reshape(X_test.shape[0], NUM_ROWS, NUM_COLS, 1)

    # Normalization
    X_train = X_train.astype(np.float32) / 255
    X_test = X_test.astype(np.float32) / 255

    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)

    # Define image size and number of classes
    image_size = X_train.shape[1:]
    classes = y_train.shape[0]

    # Create model instance and initialise Lenet Model
    lenet5 = Lenet5(image_size, classes)

    losses = []

    max_steps = X_train.shape[0] // BATCH_SIZE
    for i in range(EPOCHS):
        step = 0
        for batch_start in range(0, X_train.shape[0], BATCH_SIZE):
            if batch_start + BATCH_SIZE > X_train.shape[0]:
                continue

            inputs = X_train[batch_start:batch_start + BATCH_SIZE]
            labels = y_train[batch_start:batch_start + BATCH_SIZE]
            loss = lenet5.learn(inputs, labels)
            losses.append(loss)

            step += 1
            print(f"\rEpoch {i:4d}/{EPOCHS}: Step {step:4d}/{max_steps}: loss = {loss:.4f}", end="")


    ### test
    accuracies = []

    step = 0
    max_steps = X_test.shape[0] // BATCH_SIZE

    for batch_start in range(0, X_test.shape[0], BATCH_SIZE):
        if batch_start + BATCH_SIZE > X_test.shape[0]:
            continue

        inputs = X_test[batch_start:batch_start + BATCH_SIZE]
        labels = y_test[batch_start:batch_start + BATCH_SIZE]

        prediction = lenet5.predict(inputs)
        prediction = np.argmax(prediction, -1)
        accuracies.append(np.sum(prediction == labels) / BATCH_SIZE)

        step += 1
        print(f"\rStep {step:4d}/{max_steps}", end="")
        # print()

    tfaccuracy = np.mean(accuracies)
    print(f"\n======= tensorflow Test accuracy: {tfaccuracy:.3f}\n")
        
    ######## iree train

    exported_names = ["predict", "learn"]

    backend_choice = "llvm-cpu (CPU)" #@param [ "vmvx (CPU)", "llvm-cpu (CPU)", "vulkan-spirv (GPU/SwiftShader – requires additional drivers) " ]
    backend_choice = backend_choice.split(' ')[0]

    # Compile the Lenet5 module
    # Note: extra flags are needed to i64 demotion, see https://github.com/iree-org/iree/issues/8644
    vm_flatbuffer = iree.compiler.tf.compile_module(
        Lenet5(image_size, classes),
        target_backends=[backend_choice],
        exported_names=exported_names,
        extra_args=["--iree-mhlo-demote-i64-to-i32=false",
                    "--iree-flow-demote-i64-to-i32"])
    compiled_model = iree.runtime.load_vm_flatbuffer(
        vm_flatbuffer,
        backend=backend_choice)

    #@title Benchmark inference and training
    # infer_result = compiled_model.predict(x_train[:BATCH_SIZE])
    # train_result = compiled_model.learn(x_train[:BATCH_SIZE], y_train[:BATCH_SIZE])
    # print("Inference latency:\n  ", end="")%timeit -n 100 compiled_model.predict(x_train[:BATCH_SIZE])
    # print("Training latancy:\n  ", end="")%timeit -n 100 compiled_model.learn(x_train[:BATCH_SIZE], y_train[:BATCH_SIZE])

    # Run the core training loop.
    losses = []

    max_steps = X_train.shape[0] // BATCH_SIZE
    for i in range(EPOCHS):
        step = 0
        for batch_start in range(0, X_train.shape[0], BATCH_SIZE):
            if batch_start + BATCH_SIZE > X_train.shape[0]:
                continue

            inputs = X_train[batch_start:batch_start + BATCH_SIZE]
            labels = y_train[batch_start:batch_start + BATCH_SIZE]

            loss = compiled_model.learn(inputs, labels).to_host()
            losses.append(loss)

            step += 1
            print(f"\rEpoch {i:4d}/{EPOCHS}: Step {step:4d}/{max_steps}: loss = {loss:.4f}", end="")

    #@title Plot the training results
    # !python -m pip install bottleneck
    import bottleneck as bn
    smoothed_losses = bn.move_mean(losses, 32)
    x = np.arange(len(losses))

    # plt.plot(x, smoothed_losses, linewidth=2, label='loss (moving average)')
    # plt.scatter(x, losses, s=16, alpha=0.2, label='loss (per training step)')

    # plt.ylim(0)
    # plt.legend(frameon=True)
    # plt.xlabel("training step")
    # plt.ylabel("cross-entropy")
    # plt.title("training loss");

    #@title Evaluate the network on the test data.
    accuracies = []

    step = 0
    max_steps = X_test.shape[0] // BATCH_SIZE
    for batch_start in range(0, X_test.shape[0], BATCH_SIZE):
        if batch_start + BATCH_SIZE > X_test.shape[0]:
            continue

        inputs = X_test[batch_start:batch_start + BATCH_SIZE]
        labels = y_test[batch_start:batch_start + BATCH_SIZE]

        prediction = compiled_model.predict(inputs).to_host()
        prediction = np.argmax(prediction, -1)
        accuracies.append(np.sum(prediction == labels) / BATCH_SIZE)

        step += 1
        print(f"\rStep {step:4d}/{max_steps}", end="")

        accuracy = np.mean(accuracies)
    print(f"============ iree Test accuracy: {accuracy:.3f}")

    '''
    #@title Display inference predictions on a random selection of heldout data
    rows = 4
    columns = 4
    images_to_display = rows * columns
    assert BATCH_SIZE >= images_to_display

    random_index = np.arange(x_test.shape[0])
    np.random.shuffle(random_index)
    x_test = x_test[random_index]
    y_test = y_test[random_index]

    predictions = compiled_model.predict(x_test[:BATCH_SIZE]).to_host()
    predictions = np.argmax(predictions, -1)

    fig, axs = plt.subplots(rows, columns)

    for i, ax in enumerate(np.ndarray.flatten(axs)):
    ax.imshow(x_test[i, :, :, 0])
    color = "#000000" if predictions[i] == y_test[i] else "#ff7f0e"
    ax.set_xlabel(f"prediction={predictions[i]}", color=color)
    ax.grid(False)
    ax.set_yticks([])
    ax.set_xticks([])

    fig.tight_layout()
    '''