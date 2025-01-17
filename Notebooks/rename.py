import os
import argparse


def replace_filenames(directory, old_string, new_string):
    # List all files in the directory
    files = os.listdir(directory)

    # Iterate over each file
    for filename in files:
        # Check if the old_string is present in the filename
        if old_string in str(filename):
            print(f'filename {filename}')
            # Construct the new filename by replacing old_string with new_string
            new_filename = filename.replace(old_string, new_string)

            # Construct the full paths for the old and new filenames
            old_filepath = os.path.join(directory, filename)
            new_filepath = os.path.join(directory, new_filename)

            # Rename the file
            os.rename(old_filepath, new_filepath)
            print(f"Renamed '{filename}' to '{new_filename}'")


def main():
    # Get the current working directory
    current_directory = os.getcwd()

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Replace old string in filenames with new string.')
    parser.add_argument('old_string', metavar='OLD_STRING', type=str, help='The string to be replaced')
    parser.add_argument('new_string', metavar='NEW_STRING', type=str, help='The string to replace with')
    args = parser.parse_args()

    # Call the function to replace filenames
    replace_filenames(current_directory, args.old_string, args.new_string)


if __name__ == "__main__":
    main()
