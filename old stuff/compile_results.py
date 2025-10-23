import os
import shutil

# Global counters
missing_results_csv_count = 0
incorrect_csv_count = 0
results_csv_found_count = 0  # Tracks how many folders contain a 'Results CSV'

def compile_results():
    global missing_results_csv_count, incorrect_csv_count, results_csv_found_count
    generated_dataset_path = r'C:/Users/yansari/Desktop/Workstation/Thermal Homography/Generated Dataset'

    # Iterate through colormap folders
    for colormap_folder in os.listdir(generated_dataset_path):
        colormap_path = os.path.join(generated_dataset_path, colormap_folder)
        if not os.path.isdir(colormap_path):
            continue

        # Iterate through timing folders
        for timing_folder in os.listdir(colormap_path):
            timing_path = os.path.join(colormap_path, timing_folder)
            if not os.path.isdir(timing_path):
                continue

            # Determine if it's "with angle" or "without angle"
            if timing_folder == "June02_12pm":  # "without angle"
                print(f"Processing 'without angle' folder: {timing_folder}")
                # No angle folders, directly check distance folders
                check_and_clean_distance_folders(timing_path)
            else:  # "with angle"
                print(f"Processing 'with angle' folder: {timing_folder}")
                # Check angle folders
                for angle_folder in os.listdir(timing_path):
                    angle_path = os.path.join(timing_path, angle_folder)
                    if not os.path.isdir(angle_path):
                        continue
                    check_and_clean_distance_folders(angle_path)

    # Print final counts
    print("\nVerification Summary:")
    print(f"Total folders with 'Results CSV': {results_csv_found_count}")
    print(f"Missing 'Results CSV' folders: {missing_results_csv_count}")
    print(f"'Results CSV' folders with fewer than 20 CSV files: {incorrect_csv_count}")

def check_and_clean_distance_folders(base_path):
    global missing_results_csv_count, incorrect_csv_count, results_csv_found_count
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)

        # Remove Merged Distance Results folders
        if folder == "Merged Distance Results" and os.path.isdir(folder_path):
            print(f"Removing folder: {folder_path}")
            shutil.rmtree(folder_path)
            continue

        # Check for distance folders
        if not os.path.isdir(folder_path):
            continue

        # Check for Results CSV folder
        results_csv_path = os.path.join(folder_path, 'Results CSV')
        if os.path.isdir(results_csv_path):
            results_csv_found_count += 1  # Increment for each found 'Results CSV'
        else:
            print(f"Missing 'Results CSV' folder in: {folder_path}")
            missing_results_csv_count += 1
            continue

        # Check for 20 CSV files
        csv_files = [f for f in os.listdir(results_csv_path) if f.endswith('.csv')]
        if len(csv_files) != 20:
            print(f"Incorrect number of CSV files in: {results_csv_path}. Found: {len(csv_files)}")
            incorrect_csv_count += 1
            continue

        # Check for empty CSV files
        for csv_file in csv_files:
            csv_path = os.path.join(results_csv_path, csv_file)
            if os.path.getsize(csv_path) == 0:
                print(f"Empty CSV file found: {csv_path}")

 

def validate_colormap_folders():
    generated_dataset_path = r'C:/Users/yansari/Desktop/Workstation/Thermal Homography/Generated Dataset'

    # List to keep track of colormap folders that fail the test
    invalid_colormap_folders = []

    # Iterate through colormap folders
    for colormap_folder in os.listdir(generated_dataset_path):
        colormap_path = os.path.join(generated_dataset_path, colormap_folder)
        if not os.path.isdir(colormap_path):
            continue

        # Check if colormap folder contains at least one valid Results CSV folder
        if not has_valid_results_csv(colormap_path):
            invalid_colormap_folders.append(colormap_folder)

    # Print the result
    if invalid_colormap_folders:
        print("Colormap folders failing the test:")
        for folder in invalid_colormap_folders:
            print(f"- {folder}")
    else:
        print("All colormap folders have at least one valid 'Results CSV' folder.")

def has_valid_results_csv(base_path):
    """Recursively check if there's at least one valid 'Results CSV' folder in the subdirectories."""
    for root, dirs, files in os.walk(base_path):
        for folder in dirs:
            if folder == "Results CSV":
                results_csv_path = os.path.join(root, folder)
                csv_files = [f for f in os.listdir(results_csv_path) if f.endswith('.csv')]

                # Check if this Results CSV folder is valid (contains exactly 20 CSV files)
                if len(csv_files) == 20:
                    return True  # Valid Results CSV folder found
    return False
 

import os
import pandas as pd

def get_results_csv_paths():
    """
    Returns the path of the Results CSV per angle group (if applicable) or per timing folder 
    (if angle folder is not present).
    """
    generated_dataset_path = r'C:/Users/yansari/Desktop/Workstation/Thermal Homography/Generated Dataset'
    results_csv_paths = {}

    # Iterate through colormap folders
    for colormap_folder in os.listdir(generated_dataset_path):
        colormap_path = os.path.join(generated_dataset_path, colormap_folder)
        if not os.path.isdir(colormap_path):
            continue

        # Dictionary to store paths for this colormap
        results_csv_paths[colormap_folder] = {}

        # Iterate through timing folders
        for timing_folder in os.listdir(colormap_path):
            timing_path = os.path.join(colormap_path, timing_folder)
            if not os.path.isdir(timing_path):
                continue

            # Check if timing folder contains angle folders
            angle_group_paths = {}
            has_angle_folders = False

            for subfolder in os.listdir(timing_path):
                subfolder_path = os.path.join(timing_path, subfolder)
                if os.path.isdir(subfolder_path) and subfolder.endswith("deg"):  # Angle folder check
                    has_angle_folders = True
                    # Check for Results CSV in this angle group
                    for distance_folder in os.listdir(subfolder_path):
                        distance_path = os.path.join(subfolder_path, distance_folder)
                        results_csv_path = os.path.join(distance_path, "Results CSV")
                        if os.path.isdir(results_csv_path):
                            angle_group_paths[subfolder] = results_csv_path

            # If angle folders exist, save their paths; otherwise, check timing folder directly
            if has_angle_folders:
                results_csv_paths[colormap_folder][timing_folder] = angle_group_paths
            else:
                for distance_folder in os.listdir(timing_path):
                    distance_path = os.path.join(timing_path, distance_folder)
                    results_csv_path = os.path.join(distance_path, "Results CSV")
                    if os.path.isdir(results_csv_path):
                        results_csv_paths[colormap_folder][timing_folder] = results_csv_path

    return results_csv_paths

import os
import pandas as pd

def create_merged_csv():
    """
    For each 'Results CSV' folder, create a 'merged.csv' file with averaged results.
    """
    generated_dataset_path = r'C:/Users/yansari/Desktop/Workstation/Thermal Homography/Generated Dataset'

    # Walk through the directory structure
    for root, dirs, files in os.walk(generated_dataset_path):
        for folder in dirs:
            if folder == "Results CSV":
                results_csv_path = os.path.join(root, folder)
                process_results_folder(results_csv_path)

def process_results_folder(results_csv_path):
    """
    Process a single 'Results CSV' folder and create a merged.csv file.
    """
    # Initialize the merged data
    merged_data = []

    # Iterate through all CSV files in the folder
    for csv_file in os.listdir(results_csv_path):
        # Skip the merged.csv file
        if csv_file == "merged.csv" or not csv_file.endswith('.csv'):
            continue

        csv_path = os.path.join(results_csv_path, csv_file)
        try:
            # Read the CSV file
            df = pd.read_csv(csv_path)

            # Initialize variables for summing and counting valid entries
            mean_ace = mean_ssim = mean_mi = mean_afrr = 0
            count_ace = count_ssim = count_mi = count_afrr = 0

            # Iterate through the rows and calculate averages manually
            for _, row in df.iterrows():
                try:
                    if row.iloc[1] != "F":  # Skip "F" for ACE
                        mean_ace += float(row.iloc[1])
                        count_ace += 1
                    if row.iloc[2] != "F":  # Skip "F" for SSIM
                        mean_ssim += float(row.iloc[2])
                        count_ssim += 1
                    if row.iloc[3] != "F":  # Skip "F" for MI
                        mean_mi += float(row.iloc[3])
                        count_mi += 1
                    if row.iloc[4] != "F":  # Skip "F" for AFRR
                        mean_afrr += float(row.iloc[4])
                        count_afrr += 1
                except Exception as e:
                    print(f"Error processing row in file {csv_file}: {e}")

            # Calculate final means
            mean_ace = mean_ace / count_ace if count_ace > 0 else None
            mean_ssim = mean_ssim / count_ssim if count_ssim > 0 else None
            mean_mi = mean_mi / count_mi if count_mi > 0 else None
            mean_afrr = mean_afrr / count_afrr if count_afrr > 0 else None

            # Append the results to the merged data
            merged_data.append({
                "csv name": csv_file,
                "mean ACE": mean_ace,
                "mean SSIM": mean_ssim,
                "mean MI": mean_mi,
                "mean AFRR": mean_afrr
            })
        except Exception as e:
            print(f"Error processing file {csv_file}: {e}")

    # Create a DataFrame from the merged data
    merged_df = pd.DataFrame(merged_data)

    # Save the merged data to a new CSV file in the same folder
    merged_csv_path = os.path.join(results_csv_path, "merged.csv")
    merged_df.to_csv(merged_csv_path, index=False)
    print(f"Created merged CSV: {merged_csv_path}")

# Call the function
#create_merged_csv()


def count_failures():
    """
    Process the Results CSV folders to count instances of 'F' in each image instance
    and update the merged.csv with failure counts.
    """
    generated_dataset_path = r'C:/Users/yansari/Desktop/Workstation/Thermal Homography/Generated Dataset'

    # Walk through the directory structure
    for root, dirs, files in os.walk(generated_dataset_path):
        for folder in dirs:
            if folder == "Results CSV":
                results_csv_path = os.path.join(root, folder)
                process_failure_counts(results_csv_path)

def process_failure_counts(results_csv_path):
    """
    Process a single 'Results CSV' folder to count 'F' instances and update merged.csv.
    """
    # Load the merged.csv file
    merged_csv_path = os.path.join(results_csv_path, "merged.csv")
    if not os.path.exists(merged_csv_path):
        print(f"Missing merged.csv in {results_csv_path}")
        return

    merged_df = pd.read_csv(merged_csv_path)

    # Initialize columns for failure counts
    merged_df["25% Failure"] = 0
    merged_df["50% Failure"] = 0
    merged_df["75% Failure"] = 0
    merged_df["100% Failure"] = 0

    # Iterate through all CSV files in the folder
    for csv_file in os.listdir(results_csv_path):
        # Skip merged.csv
        if csv_file == "merged.csv" or not csv_file.endswith('.csv'):
            continue

        csv_path = os.path.join(results_csv_path, csv_file)
        try:
            # Read the CSV file
            df = pd.read_csv(csv_path)

            # Initialize failure counts
            counts = {1: 0, 2: 0, 3: 0, 4: 0}

            # Process each row (image instance)
            for _, row in df.iterrows():
                # Count the number of 'F' instances in the row (columns 2 to 5)
                f_count = row.iloc[1:5].value_counts().get("F", 0)
                if f_count > 0:
                    counts[f_count] += 1

            # Update the merged.csv file with failure counts for this CSV
            merged_df.loc[merged_df["csv name"] == csv_file, "25% Failure"] = counts[1]
            merged_df.loc[merged_df["csv name"] == csv_file, "50% Failure"] = counts[2]
            merged_df.loc[merged_df["csv name"] == csv_file, "75% Failure"] = counts[3]
            merged_df.loc[merged_df["csv name"] == csv_file, "100% Failure"] = counts[4]

            # Print failure counts for this file
            print(f"File: {csv_file}, 25%: {counts[1]}, 50%: {counts[2]}, 75%: {counts[3]}, 100%: {counts[4]}")
        except Exception as e:
            print(f"Error processing file {csv_file}: {e}")

    # Save the updated merged.csv
    merged_df.to_csv(merged_csv_path, index=False)
    print(f"Updated merged.csv with failure counts: {merged_csv_path}")

# Call the function
#count_failures()

import os
import pandas as pd

def append_total_samples():
    """
    Append a column 'total_samples' to merged.csv files indicating the total number of rows
    (excluding headers) for each CSV file.
    """
    generated_dataset_path = r'C:/Users/yansari/Desktop/Workstation/Thermal Homography/Generated Dataset'

    # Walk through the directory structure
    for root, dirs, files in os.walk(generated_dataset_path):
        for folder in dirs:
            if folder == "Results CSV":
                results_csv_path = os.path.join(root, folder)
                update_total_samples(results_csv_path)

def update_total_samples(results_csv_path):
    """
    Update the merged.csv file in the given Results CSV folder with the 'total_samples' column.
    """
    # Load the merged.csv file
    merged_csv_path = os.path.join(results_csv_path, "merged.csv")
    if not os.path.exists(merged_csv_path):
        print(f"Missing merged.csv in {results_csv_path}")
        return

    merged_df = pd.read_csv(merged_csv_path)

    # Initialize the total_samples column
    merged_df["total_samples"] = 0

    # Iterate through all CSV files in the folder
    for csv_file in os.listdir(results_csv_path):
        # Skip merged.csv
        if csv_file == "merged.csv" or not csv_file.endswith('.csv'):
            continue

        csv_path = os.path.join(results_csv_path, csv_file)
        try:
            # Count the total rows (excluding header)
            total_rows = sum(1 for line in open(csv_path)) - 1  # Subtract 1 for the header
            # Update the merged.csv file with the total row count for this CSV
            merged_df.loc[merged_df["csv name"] == csv_file, "total_samples"] = total_rows
        except Exception as e:
            print(f"Error processing file {csv_file}: {e}")

    # Save the updated merged.csv
    merged_df.to_csv(merged_csv_path, index=False)
    print(f"Updated merged.csv with total_samples: {merged_csv_path}")

# Call the function
append_total_samples()

