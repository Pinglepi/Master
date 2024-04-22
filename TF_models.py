from keras.models import Sequential
from keras.layers import Dense, Flatten, Input, Conv1D, BatchNormalization, ReLU, GlobalAveragePooling1D, Dropout, MaxPooling1D, AveragePooling1D
from keras.optimizers import Adam
from keras.regularizers import l2
import xgboost as xgb
from TF_models import *
from keras.callbacks import EarlyStopping

from tmu.models.classification.vanilla_classifier import TMClassifier
import numpy as np


# Define DNN model creation function
def Create_DNN(X_train, y_train, X_val, y_val):
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_val = X_val.reshape(X_val.shape[0], -1)
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(patience=15, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=100, batch_size=256, validation_data=(X_val, y_val), callbacks=[early_stopping])
    
    return model

# Define CNN model creation function
def Create_CNN_simple(X_train, y_train, X_val, y_val):
    model = Sequential()
    model.add(Conv1D(32, 3, activation='relu', input_shape=(X_train.shape[1], 20)))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(patience=15, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=100, batch_size=256, validation_data=(X_val, y_val), callbacks=[early_stopping])
    
    return model

def Create_CNN_super(X_train, y_train, X_val, y_val):
    model = Sequential()
    
    # Convolutional layer with L2 regularization
    model.add(Conv1D(64, 3, activation='relu', input_shape=(200, 20), kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))

    # Second convolutional layer
    model.add(Conv1D(128, 3, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))

    # Third convolutional layer
    model.add(Conv1D(256, 3, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))

    # Flatten and Dense layers with dropout and L2 regularization
    model.add(Flatten())
    model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.5))  # Increased dropout to prevent overfitting
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model with learning_rate instead of lr
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(patience=15, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=100, batch_size=256, validation_data=(X_val, y_val), callbacks=[early_stopping])
    
    return model

def Create_XGBoost(X_train, y_train, X_val, y_val):
    # Reshape data to 2D
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_val = X_val.reshape(X_val.shape[0], -1)
    
    # Create DMatrix objects for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_val, label=y_val)
    
    # Specify the number of training rounds
    num_boost_round = 1000

    # Train the model with early stopping
    model = xgb.train(
        {'eval_metric': 'logloss'},  # Set the evaluation metric for validation
        dtrain, 
        num_boost_round, 
        evals=[(dtest, 'test')], 
        early_stopping_rounds=15
    )

    return model

def batch_generator(data, targets, batch_size, shuffle=True):
    """Generate batches of data along with their corresponding target values."""
    num_samples = len(data)
    indices = np.arange(num_samples)
    if shuffle:
        np.random.shuffle(indices)
    num_batches = num_samples // batch_size
    for i in range(num_batches):
        start_index = i * batch_size
        end_index = (i + 1) * batch_size
        batch_indices = indices[start_index:end_index]
        batch_data = data[batch_indices]
        batch_targets = targets[batch_indices]
        yield batch_data, batch_targets

def Create_tsetlin(X_train, y_train, X_val, y_val):
    X_train = X_train.reshape(X_train.shape[0], -1)
    C = 10000
    tm = TMClassifier(number_of_clauses=C,
        T=np.sqrt(C/2)+2,
        s=2.534 * np.log(C/3.7579),
        platform="GPU",
        weighted_clauses=True,)

    # Example usage for training
    batch_size = 512  # Set desired batch size
    num_epochs = 1 # Set the number of epochs
    total_samples = len(X_train)
    num_batches_per_epoch = total_samples // batch_size

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        for batch_index, (batch_data, batch_targets) in enumerate(batch_generator(X_train, y_train, batch_size)):
            tm.fit(batch_data, batch_targets)
            percent_complete = (batch_index + 1) / num_batches_per_epoch * 100
            print(f"\rProcessing batch {batch_index + 1}/{num_batches_per_epoch} ({percent_complete:.2f}%)", end="")
    return tm

def Create_tsetlin_convolution(X_train, y_train, X_val, y_val):
    C = 10000
    tm = TMClassifier(number_of_clauses=C,
        T=np.sqrt(C/2)+2,
        s=2.534 * np.log(C/3.7579),
        platform="GPU",
        weighted_clauses=True,
        patch_dim=(3, 20))

    # Example usage for training
    batch_size = 512  # Set desired batch size
    num_epochs = 1 # Set the number of epochs
    total_samples = len(X_train)
    num_batches_per_epoch = total_samples // batch_size

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        for batch_index, (batch_data, batch_targets) in enumerate(batch_generator(X_train, y_train, batch_size)):
            tm.fit(batch_data, batch_targets)
            percent_complete = (batch_index + 1) / num_batches_per_epoch * 100
            print(f"\rProcessing batch {batch_index + 1}/{num_batches_per_epoch} ({percent_complete:.2f}%)", end="")
    return tm