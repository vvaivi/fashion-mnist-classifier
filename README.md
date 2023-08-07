# Fashion MNIST classifier

CNN classifier trained on the MNIST Fashion dataset. After the model is trained with clear images, the classification accuracy is evaluated on clear images, noisy images and images denoised with a CNN autoencoder. Finally, the model is trained with noisy images and its accuracy evaluated on them again to get a reference on the performance. 

## Running the code

To locally run the code, please clone this repository and execute the shell script inside the project folder. This will create a container with required packages installed. The code is mounted inside the container, so if you wish to make modifications to the code and run it again, it can be done for example with command 

### `docker start -ai mnist-classifier-container`
