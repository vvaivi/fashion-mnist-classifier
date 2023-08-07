import tensorflow as tf
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

# Add noise to images
noise_factor = 0.2
train_images_noisy = train_images + noise_factor * tf.random.normal(shape=train_images.shape)
test_images_noisy = test_images + noise_factor * tf.random.normal(shape=test_images.shape)

train_images_noisy = tf.clip_by_value(train_images_noisy,
clip_value_min=0., clip_value_max=1.)
test_images_noisy = tf.clip_by_value(test_images_noisy,
clip_value_min=0., clip_value_max=1.)

# Plot clear and noisy images for comparison
n = 2
plt.figure(figsize=(4, 4))
for i in range(n):
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(test_images[i])
    plt.title("Clear")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(test_images_noisy[i])
    plt.title("Noisy")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.savefig('plot_clear_noisy.png')

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), strides=1, activation='relu', input_shape=(28, 28, 1)),  
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax'),
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# Train the model with clean images
model.fit(train_images, train_labels, epochs=3)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy with clean images:', test_acc)
print()

test_loss_noisy, test_acc_noisy = model.evaluate(test_images_noisy, test_labels, verbose=2)
print('\nTest accuracy with noisy images:', test_acc_noisy)
print()

# Define structure for denoising
class Denoise(Model):
    def __init__(self):
        super(Denoise, self).__init__()
        self.encoder = tf.keras.Sequential([
          tf.keras.layers.Input(shape=(28, 28, 1)),
          tf.keras.layers.Conv2D(8, (3, 3), activation='relu', strides=1, padding='same')])

        self.decoder = tf.keras.Sequential([
          tf.keras.layers.Conv2DTranspose(8, kernel_size=3, strides=1, activation='relu', padding='same'),
          tf.keras.layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

autoencoder = Denoise()

autoencoder.compile(optimizer='adam',
              loss=tf.keras.losses.MeanSquaredError())

# Train autoencoder
autoencoder.fit(train_images_noisy, train_images,
                epochs=1,
                shuffle=True,
                validation_data=(test_images_noisy, test_images))

# Plot examples of denoised images versus noisy images
encoded_imgs = autoencoder.encoder(test_images_noisy).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

n = 4
plt.figure(figsize=(8, 4))
for i in range(n):
    ax = plt.subplot(2, n, i + 1)
    plt.title("Noisy")
    plt.imshow(tf.squeeze(test_images_noisy[i]))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    bx = plt.subplot(2, n, i + n + 1)
    plt.title("Denoised")
    plt.imshow(tf.squeeze(decoded_imgs[i]))
    plt.gray()
    bx.get_xaxis().set_visible(False)
    bx.get_yaxis().set_visible(False)
plt.savefig('plot_noisy_denoised.png')

denoised_loss, denoised_acc = model.evaluate(decoded_imgs, test_labels)
print('\nTest accuracy with denoised images:', denoised_acc)
print()

# Train model with noisy images
model.fit(train_images_noisy, train_labels, epochs=3)

# Evaluate the model on noisy test set
test_loss_noisy, test_acc_noisy = model.evaluate(test_images_noisy, test_labels, verbose=2)
print('\nTest accuracy with noisy images:', test_acc_noisy)