import os
import numpy as np
from preprocess import read_protocol, preprocess_data, normalize_data
from rnn_model import build_rnn_model
from cnn_model import build_cnn_model

data_dir = 'data\\LA\\LA'
train_protocol = read_protocol(os.path.join(data_dir, 'ASVspoof2019_LA_cm_protocols', 'ASVspoof2019.LA.cm.train.trn.txt'), os.path.join(data_dir, 'ASVspoof2019_LA_train'))
dev_protocol = read_protocol(os.path.join(data_dir, 'ASVspoof2019_LA_cm_protocols', 'ASVspoof2019.LA.cm.dev.trl.txt'), os.path.join(data_dir, 'ASVspoof2019_LA_dev'))
eval_protocol = read_protocol(os.path.join(data_dir, 'ASVspoof2019_LA_cm_protocols', 'ASVspoof2019.LA.cm.eval.trl.txt'), os.path.join(data_dir, 'ASVspoof2019_LA_eval'))

X_train, y_train = preprocess_data(train_protocol)
X_dev, y_dev = preprocess_data(dev_protocol)
X_eval, y_eval = preprocess_data(eval_protocol)

# Normalize data
mean = np.mean(X_train, axis=(0, 1))
std = np.std(X_train, axis=(0, 1))

X_train = (X_train - mean) / std
X_dev = (X_dev - mean) / std
X_eval = (X_eval - mean) / std

# Save mean and std for later use
np.save('./models/mean.npy', mean)
np.save('./models/std.npy', std)

# RNN model training
rnn_model = build_rnn_model((X_train.shape[1], X_train.shape[2]))
rnn_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_dev, y_dev))
rnn_model.save('./models/rnn_model.h5')

# CNN model training
X_train_cnn = X_train[..., np.newaxis]
X_dev_cnn = X_dev[..., np.newaxis]
X_eval_cnn = X_eval[..., np.newaxis]

cnn_model = build_cnn_model((X_train_cnn.shape[1], X_train_cnn.shape[2], 1))
cnn_model.fit(X_train_cnn, y_train, epochs=20, batch_size=32, validation_data=(X_dev_cnn, y_dev))
cnn_model.save('./models/cnn_model.h5')
