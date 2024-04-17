import numpy as np

def one_hot_encode(sequence, padding, standard_amino_acids='ARNDCQEGHILKMFPSTWYV'):
    """One-hot encode a single amino acid sequence using a binary array, padding in front."""
    # Use dtype=np.bool_ for memory efficiency
    encoding = np.zeros((padding, 20), dtype=np.bool_)
    # Calculate starting position for encoding to ensure front padding
    start_pos = max(0, padding - len(sequence))
    
    for index, amino_acid in enumerate(sequence):
        if index + start_pos == padding:
            break  # Stop encoding if the end of the padding is reached
        try:
            aa_index = standard_amino_acids.index(amino_acid)
            encoding[start_pos + index, aa_index] = True
        except ValueError:
            # Non-standard amino acid found, return None to indicate failure
            return None
    return encoding


def encode_batch(df, padding, sequence_column, target_column, standard_amino_acids):
    encoded_sequences = []
    valid_indices = []

    # Encode sequences in the batch
    for i, sequence in enumerate(df[sequence_column]):
        if set(sequence).issubset(set(standard_amino_acids)):
            encoded = one_hot_encode(sequence=sequence, padding=padding)
            if encoded is not None:
                encoded_sequences.append(encoded)
                valid_indices.append(i)

    # Filter the target column based on valid indices in the batch
    filtered_targets = df.iloc[valid_indices][target_column].values

    return encoded_sequences, filtered_targets


def one_hot_encode_sequences_in_batches(df, batch_size, padding, sequence_column="Sequence", target_column="has_hmm"):
    standard_amino_acids = 'ARNDCQEGHILKMFPSTWYV'
    """
    One-hot encodes amino acid sequences in a specified column of a pandas DataFrame and filters the target column accordingly.
    - Sequences containing non-standard amino acids are disregarded.
    - Sequences are padded with zeros to reach a specified length or truncated.
    - Returns a three-dimensional numpy array with shape (valid_sequences, sequence_length, 20) and the filtered target column.
    """
    # Initialize lists to store encoded sequences and filtered targets
    encoded_sequences = []
    filtered_targets = []

    # Process sequences in batches
    for start_index in range(0, len(df), batch_size):
        end_index = min(start_index + batch_size, len(df))
        batch_df = df.iloc[start_index:end_index]

        batch_encoded_sequences, batch_filtered_targets = encode_batch(batch_df, padding, sequence_column, target_column, standard_amino_acids)

        # Append results to lists
        encoded_sequences.extend(batch_encoded_sequences)
        filtered_targets.extend(batch_filtered_targets)

    # Convert lists to numpy arrays
    encoded_sequences = np.array(encoded_sequences)
    filtered_targets = np.array(filtered_targets)

    return encoded_sequences, filtered_targets