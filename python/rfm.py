import torch

def read_rfm_atm(filename):
    data = {}
    with open(filename, "r") as f:
        lines = f.readlines()

    n_levels = int(lines[0].strip())  # First line is number of levels

    current_key = None
    current_values = []

    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue

        if line.startswith("*"):
            if current_key is not None:
                # Save the previous section
                value = torch.tensor(current_values, dtype=torch.float32)
                data[current_key] = value
                current_values = []

            if line == "*END":
                break

            current_key = line.split()[0][1:]  # remove '*'
        else:
            current_values.extend(map(float, line.split()))

    # Save the last variable block if not already saved
    if current_key is not None and current_key not in data:
        value = torch.tensor(current_values, dtype=torch.float32)
        data[current_key] = value

    return data
