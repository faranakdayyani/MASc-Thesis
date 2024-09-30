import pandas as pd
import math
import numpy as np
import torch
from torch import nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F


def get_complete_date_range(df, freq):
    actual_start_date = df.index.min()
    actual_end_date = df.index.max()
    extended_start_date = (actual_start_date - pd.Timedelta(days=actual_start_date.weekday())).normalize()
    extended_end_date = actual_end_date + pd.Timedelta(days=(6 - actual_end_date.weekday()))
    extended_end_date = pd.Timestamp(extended_end_date.date()) + pd.Timedelta(days=1, seconds=-1)
    complete_date_range = pd.date_range(start=extended_start_date, end=extended_end_date, freq=freq)
    return complete_date_range


def haversine(lat1, lon1, lat2, lon2):
    # Radius of the Earth in kilometers
    R = 6371.0

    # Convert latitude and longitude from degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Difference in coordinates
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Haversine formula
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c

    return distance


def get_maximum_distance(df_segment, home_lat, home_lon):
    max_distance = 0
    for index, row in df_segment.iterrows():
        lat, lon = row['latitude'], row['longitude']
        distance = haversine(home_lat, home_lon, lat, lon)
        max_distance = int(max(max_distance, distance))
    return max_distance


def get_travelled_distance(df_segment, home_lat, home_lon):

    prev_lat, prev_lon = home_lat, home_lon

    travelled_distance = 0
    for index, row in df_segment.iterrows():
        lat, lon = row['latitude'], row['longitude']
        distance = haversine(prev_lat, prev_lon, lat, lon)
        travelled_distance += distance
        prev_lat, prev_lon = lat, lon
    return travelled_distance


def get_time_difference(data):
    if data.shape[0] == 0:
        return np.NaN
    else:
        time_difference = int((pd.to_datetime(data.index[-1]) - pd.to_datetime(data.index[0])).total_seconds() / 60)
        # print(data.shape[0], data.index[0], data.index[-1], type(time_difference))
        return time_difference


def get_nonzero_daytime_ratio(data):
    # value = data[data != 0].shape[0] / data.shape[0]
    morning_index = 8
    night_index = 20
    data = data[morning_index:night_index]
    value = data[data != 0].shape[0] / data.shape[0]

    if pd.isna(value):
        return np.NaN
    else:
        return round(value, 2)


def get_nonzero_mean(data):
    value = data[data != 0].mean()

    if pd.isna(value):
        return np.NaN
    else:
        return round(value, 2)


def get_nonzero_count(data):
    value = data[data != 0].shape[0]

    if pd.isna(value):
        return np.NaN
    else:
        return round(value, 2)


def get_nonzero_std(data):
    value = data[data != 0].std()

    if pd.isna(value):
        return np.NaN
    else:
        return round(value, 2)


def get_nonzero_min(data):
    value = data[data != 0].min()

    if pd.isna(value):
        return np.NaN
    else:
        return round(value, 2)


def get_max_timestamp(data):
    value = data.argmax()

    if pd.isna(value):
        return np.NaN
    else:
        return int(value)


def normalize_multivariate(padded_sequences, lengths):
    lengths = lengths.numpy()
    padded_sequences = padded_sequences.numpy()

    maxLength = np.max(lengths)
    num_features = padded_sequences.shape[2]  # Assuming signals.shape = [num_sequences, max_time_steps, num_features]
    newSignals = []

    for i in range(len(padded_sequences)):
        # Extract the valid part of the current sequence up to `lengths[i]`
        x = padded_sequences[i, :lengths[i], :]

        # Normalize each feature across the time steps of this sequence
        normalized_features = []
        for feature in range(num_features):
            feature_column = x[:, feature]
            normalized_feature = (feature_column - np.min(feature_column)) / np.ptp(feature_column)
            normalized_features.append(normalized_feature)

        # Stack the normalized features back into a 2D array [time_steps, features]
        normalized_x = np.stack(normalized_features, axis=1)

        # Resize the sequence to have maxLength, padding with zeros
        # This creates a new array with shape [maxLength, num_features] and copies normalized_x into it
        padded_x = np.zeros((maxLength, num_features))
        padded_x[:len(normalized_x), :] = normalized_x

        newSignals.append(padded_x)

    # Convert the list of arrays into a 3D numpy array [num_sequences, max_time_steps, num_features]
    newSignals = np.stack(newSignals)
    return newSignals


def _shift_(arr, shift, lengths):
    shifted_arr = torch.zeros_like(arr)  # Initialize an array of zeros with the same shape

    for i, length in enumerate(lengths):
        if length > 0:  # If there's actual data to shift
            effective_shift = shift % length.item()  # Adjust shift if it exceeds the length
            non_padded_row = arr[i, :length]  # Extract non-padded data
            # Apply circular shift to the non-padded data
            shifted_non_padded_row = torch.roll(non_padded_row, shifts=effective_shift, dims=1)
            # Place shifted data back, leaving the rest as padding
            shifted_arr[i, :length] = shifted_non_padded_row

    return shifted_arr


def _slice_(arr, num_segments, lengths, labels):
    _ = np.zeros((arr.shape[0] * 2, arr.shape[1], arr.shape[2]))
    new_lengths = np.zeros((lengths.shape[0] * 2))
    new_labels = np.zeros((labels.shape[0] * 2))
    j = -1

    for i, length in enumerate(lengths):

        if num_segments == 2:

            if length > 0:
                l1 = length // num_segments
                l2 = length - l1

                j += 1
                _[j, :l1, :] = arr[i, :l1]
                new_lengths[j] = l1
                new_labels[j] = labels[i]

                j += 1
                _[j, :length - l1, :] = arr[i, l1:length]
                new_lengths[j] = length - l1
                new_labels[j] = labels[i]

    return torch.tensor(_), torch.tensor(new_lengths), torch.tensor(new_labels)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, inputs):
        """Inputs have to have dimension (batch_size, num_features, seq_length)"""
        inputs = inputs.permute(0, 2, 1)
        y1 = self.tcn(inputs)  # input should have dimension (batch_size, num_features, seq_length)
        out = self.linear(y1[:, :, -1])
        # return F.log_softmax(out, dim=1)
        return out
