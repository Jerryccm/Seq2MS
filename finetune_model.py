import argparse
import ast
import sys
from utils import (
    embed_maxquant,
    spectrum2vector,
    readmgf,
    masked_spectral_distance,
    asnp32,
    BIN_SIZE,
    MAX_PEPTIDE_LENGTH,
)
import tensorflow as tf
import pandas as pd
from tensorflow import keras as k
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--new_data", type=str, default="new_finetune_data.pkl")
parser.add_argument("--model", type=str, default="pretrained_model")
parser.add_argument("--output", type=str, default="finetuned_model_result.pkl")
parser.add_argument(
    "--mod", type=str, help="A dictionary argument or txt file", required=False
)

args = parser.parse_args()
try:
    if args.mod:
        if args.mod.endswith("}"):
            new_mod = ast.literal_eval(args.mod)
            mods.update(new_mod)
        else:
            with open(args.mod, "r") as file:
                first_line = file.readline()
            new_mod = ast.literal_eval(first_line)
            mods.update(new_mod)
except Exception as e:
    print("Error occurred while handling modifications: Check input string / file", e)
    sys.exit(1)

try:
    if args.new_data.endswith(".pkl"):
        data = pd.read_pickle(args.new_data)
    elif args.new_data.endswith(".mgf"):
        data = readmgf(args.new_data)
    else:
        data = pd.read_csv(args.new_data, sep="\t")
except Exception as e:
    print("Error occurred while loading or preprocessing the input data:", e)
    sys.exit(1)

# Remove X, U, O from sequence, because our model doesnt support
data = pd.DataFrame(data)
data = data[data["Sequence"].str.contains("X") == False]
data = data[data["Sequence"].str.contains("U") == False]
data = data[data["Sequence"].str.contains("O") == False]
print(data.head())
data = data.to_dict("Records")
print("Length of input:", len(data))

# load existing model here
model_name = args.model
pm = k.models.load_model(
    model_name, custom_objects={"masked_spectral_distance": masked_spectral_distance}
)
pm.compile(
    optimizer=k.optimizers.Adam(
        lr=0.00005
    ),  # TODO: change the lr here during your experimentation
    loss=masked_spectral_distance,
    metrics=[tf.keras.metrics.CosineSimilarity(axis=1), masked_spectral_distance],
)


class DataGen:
    def __init__(self, data, batch_size=64, shuffle=True):
        self.data = np.array(data)
        self.indexes = np.arange(self.data.shape[0])
        self.batch_size = batch_size

    def __len__(self):
        return self.data.shape[0] // self.batch_size

    def get_item(self, i):
        embed = asnp32(
            embed_maxquant(self.data[i], fixedaugment=False, key="N", havelong=True)
        )
        spectra = asnp32(
            spectrum2vector(
                self.data[i]["mz"], self.data[i]["it"], BIN_SIZE, self.data[i]["Charge"]
            )
        )
        return embed, spectra

    def __call__(self):
        for i in self.indexes:
            yield self.get_item(i)


batch_size = 64
train_gen = DataGen(data)
types = (tf.float32, tf.float16)
shapes = ((MAX_PEPTIDE_LENGTH + 2, 12), (20000))
dataset = tf.data.Dataset.from_generator(
    train_gen, output_types=types, output_shapes=shapes
)
dataset = dataset.batch(batch_size)  # this will be the training dataset

# TODO: get testing datasets, swap with your new files!
# currently im using the original test set,
# this is to track whether the model still performs well when adapting to the new finetune data
test_data = readmgf("hcd_testingset.mgf")
test_spectras = asnp32(
    [spectrum2vector(i["mz"], i["it"], BIN_SIZE, i["Charge"]) for i in test_data]
)
test_labels = asnp32([embed_maxquant(seq, augment=False) for seq in test_data])

finetune_test_data = data[
    -100:
]  # TODO: this is wrong, i just put here as place holder. Please adapt to your new finetune data
finetune_test_dataset = tf.data.Dataset.from_generator(
    DataGen(finetune_test_data), output_types=types, output_shapes=shapes
)
finetune_test_dataset = finetune_test_dataset.batch(batch_size)


print("Constructed Dataset")


def scheduler(epoch, lr):
    if epoch < 5:
        return lr
    elif epoch > 10:
        return lr
    else:
        return lr * tf.math.exp(-0.05)


callback = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=2)

checkpoint_filepath = "./checkpoint"
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath, monitor="loss", mode="min", save_best_only=True
)


class TestCallback(tf.keras.callbacks.Callback):
    def __init__(self, x, y, new_data):
        self.old_data_x = x
        self.old_data_y = y
        self.new_data = new_data

    def on_epoch_end(self, epoch, logs):
        print("New data:")
        logs["new_loss"], logs["new_sim"], logs["new_angle"] = self.model.evaluate(
            x=self.new_data, verbose=2
        )
        print("hcd:")
        logs["hcd_loss"], logs["hcd_sim"], logs["hcd_angle"] = self.model.evaluate(
            x=self.old_data_x, y=self.old_data_y, verbose=2
        )


history = pm.fit(
    x=dataset,
    epochs=10,  # TODO: play with different epoch number
    batch_size=32,  # TODO: play with different batch sizes
    callbacks=[
        callback,
        model_checkpoint_callback,
        TestCallback(
            test_labels,
            test_spectras,
            finetune_test_dataset,  # TODO: change the test datasets here
        ),
    ],
    verbose=1,
)
hist_df = pd.DataFrame(history.history)

# this saves the loss and accuracy history to a df <output directory>.pkl file, useful for plotting graphs
hist_df.to_pickle(args.output)

# this saves the finetuned model to the <output> directory
pm.save(args.output.split(".")[0])
