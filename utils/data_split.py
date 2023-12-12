def split_file(input_file, output_prefix, lines_per_file=1000):
    with open(input_file, 'r') as infile:
        lines = infile.readlines()

    total_lines = len(lines)
    num_files = (total_lines + lines_per_file - 1) // lines_per_file

    for i in range(num_files):
        end_idx = (i + 1) * lines_per_file
        chunk = lines[:end_idx]

        output_file = f"{output_prefix}_{end_idx}.csv"
        with open(output_file, 'w') as outfile:
            outfile.writelines(chunk)


if __name__ == "__main__":
    # input_file_path = "data/nus_wide_train_guest_1000.csv"
    # output_file_prefix = "data/nus_wide_train_guest"
    # split_file(input_file_path, output_file_prefix, 50)

    input_file_path = "data/nus_wide_train_host_100.csv"
    output_file_prefix = "data/nus_wide_train_host"
    split_file(input_file_path, output_file_prefix, 5)
