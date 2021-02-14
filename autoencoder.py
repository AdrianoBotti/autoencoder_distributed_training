import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import datetime
import datasetpreprocessing as dpp
import pandas as pd
import math

# Set TF_CONFIG
os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ["10.0.2.4:1717", "10.0.2.5:1818"]
    },
    'task': {'type': 'worker', 'index': 0}
})


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Encoder(tf.keras.layers.Layer):
    """Maps MNIST digits to a triplet (z_mean, z_log_var, z)."""

    def __init__(self, latent_dim=128, intermediate_dim=256, name="encoder", **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.dense_proj = layers.Dense(intermediate_dim, activation=tf.nn.relu)
        self.dense_mean = layers.Dense(latent_dim)
        self.dense_log_var = layers.Dense(latent_dim)
        self.sampling = Sampling()

    def call(self, inputs):
        x = self.dense_proj(inputs)
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z

class Decoder(tf.keras.layers.Layer):
    """Converts z, the encoded digit vector, back into a readable digit."""

    def __init__(self, original_dim, intermediate_dim=64, name="decoder", **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.dense_proj = layers.Dense(intermediate_dim, activation=tf.nn.relu)
        self.dense_output = layers.Dense(original_dim, activation=tf.nn.sigmoid)

    def call(self, inputs):
        x = self.dense_proj(inputs)
        return self.dense_output(x)


class VariationalAutoEncoder(tf.keras.Model):
    """Combines the encoder and decoder into an end-to-end model for training."""

    def __init__(self, original_dim, intermediate_dim=64, latent_dim=32, name="autoencoder", **kwargs):
        super(VariationalAutoEncoder, self).__init__(**kwargs)
        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim=latent_dim, intermediate_dim=intermediate_dim)
        self.decoder = Decoder(original_dim, intermediate_dim=intermediate_dim)

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        # Add KL divergence regularization loss.
        kl_loss = -0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
        )
        self.add_loss(kl_loss)
        return reconstructed


# Prepare directories for logs and checkpoint
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
checkpoint_dir = "ckpt/"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

def make_or_restore_model():
    # Either restore the latest model, or create a fresh one
    # if there is no checkpoint available.
    checkpoint = [checkpoint_dir + name for name in os.listdir(checkpoint_dir)]
    if checkpoint:
        print("Restoring from checkpoint")
        return get_compiled_model(checkpoint_dir)
    print("Creating a new model")
    return get_compiled_model()

def get_compiled_model(checkpoint=None):
    # Loss and optimizer.
    loss_fn = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    vae = VariationalAutoEncoder(original_dim=689, intermediate_dim=256, latent_dim=128)
    if(checkpoint):
        vae.load_weights(checkpoint)

    vae.compile(optimizer, loss_fn)
    return vae


# Prepare a dataset.
def mnist_dataset(batch_size):
    # Prepare a dataset.
    (x_train, _), _ = tf.keras.datasets.mnist.load_data()
    dataset = tf.data.Dataset.from_tensor_slices(
        x_train.reshape(60000, 784).astype("float32") / 255
    )
    dataset = dataset.map(lambda x: (x, x))  # Use x_train as both inputs & targets
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

    #disable autosharding
    #options = tf.data.Options()
    #options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    #dataset_no_auto_shard = dataset.with_options(options)

    return dataset

# prepare METABRIC dataset
def metabric_dataset(batch_size, maxtrainsize, maxvalsize):

    df = dpp.preprocess_dataset("data/METABRIC_RNA_Mutation.csv")

    dataset = tf.data.Dataset.from_tensor_slices((df.values, df.values))
    dataset = dataset.shuffle(buffer_size=2048)

    dataset_size = len(dataset)
    train_size = int(0.7 * dataset_size)
    val_size = int(0.15 * dataset_size)
    test_size = int(0.15 * dataset_size)

    #repeat trainset to allow multiple epochs
    train_rep = int( math.ceil(maxtrainsize/train_size) )
    train_set = dataset.take(train_size).repeat(train_rep).batch(batch_size)

    #repeat valset to allow multiple epochs
    val_rep = int( math.ceil(maxvalsize/val_size) )
    validation_set =dataset.skip(train_size).take(test_size).repeat(val_rep).batch(batch_size)

    test_set = dataset.skip(train_size + test_size).batch(batch_size)

    return train_set, validation_set, test_set, df


def build_classifier_dataset(encoder, dataframe):

	inputs = tf.data.Dataset.from_tensor_slices((dataframe.values)).batch(1)

	iter_inputs = iter(inputs)
	row_index = 0
	latent_list = []

	while row_index < len(inputs):

		input = next(iter_inputs)
		_, _, latent_input = encoder(input)
		array = latent_input.numpy()[0]
		latent_list.append(array.tolist())
		row_index+=1

	# convert the dataset into a dataframe
	latent_df = pd.DataFrame(latent_list)

	#add class attribute to dataframe
	targets = dataframe.pop("type_of_breast_surgery").tolist()
	latent_df["type_of_breast_surgery"] = targets
	
	print(latent_df)

	#save latent dataframe as csv
	latent_df.to_csv("data/metabric_latent.csv", index=False)

	#latent_input = encoder(inputs)



def main():
	#code to execute when lauching the script
	strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
	print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

	# Open a strategy scope and create/restore the model
	with strategy.scope():
		vae = make_or_restore_model()

	callbacks = [
		# This callback saves a SavedModel every epoch
		tf.keras.callbacks.ModelCheckpoint(
			filepath=checkpoint_dir,
			save_weights_only=True,
			monitor='loss',
			mode='min',
			save_best_only=True,
			save_freq='epoch'),
		tf.keras.callbacks.TensorBoard(log_dir)
	]

	epochs=10
	steps_per_epoch=700
	validation_steps = 300
	global_batch_size = 4

	train_set, validation_set, test_set, dataframe = metabric_dataset(
		batch_size=global_batch_size,
		maxtrainsize=epochs*steps_per_epoch*global_batch_size,
		maxvalsize=epochs*validation_steps*global_batch_size
	)

	vae.fit(train_set, epochs=epochs, 
		validation_data = validation_set,
		callbacks=callbacks, verbose=1, 
		steps_per_epoch=steps_per_epoch, validation_steps=validation_steps
	)

	result = vae.evaluate(test_set, steps=len(test_set))
	print(result)

	#creating a dataset to train the classifier on inputs from latent space
	latent_dataset = build_classifier_dataset(vae.encoder, dataframe)


if __name__ == '__main__':
    main()


