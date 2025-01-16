import numpy as np
import pandas as pd


mapper = {
    'A': 0,
    'C': 1,
    'U': 2,
    'G': 3
}

all_sequences = []
with open("dataset.txt", "r") as f:
    lines = f.readlines()

    for i in range(0, len(lines), 2):
        fasta = lines[i+1].strip()
        mapped_sequence = [mapper[c] for c in fasta]  # Map sequence using `mapper`
        sequence_array = np.array(mapped_sequence)  # Convert to numpy array
        all_sequences.append(sequence_array)  # Add to list
        print(f"Processed sequence {i // 2}")

concatenated_sequences = np.stack(all_sequences, axis=0)  # Use stack to preserve shape consistency

np.savetxt("word_embedding.csv", concatenated_sequences, delimiter=",", fmt="%d")  # Save to CSV
print(f"All sequences saved to word_embedding.csv")

