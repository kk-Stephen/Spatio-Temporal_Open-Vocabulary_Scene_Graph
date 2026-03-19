import os
import json


def count_edge_types(directory_path):
    """
    Counts the total number of spatial and temporal edges across all JSON files in a directory.

    Args:
        directory_path (str): The path to the directory containing the JSON files.

    Returns:
        tuple: A tuple containing the total count of spatial edges and temporal edges.
               Returns (0, 0) if the directory doesn't exist.
    """
    total_spatial_edges = 0
    total_temporal_edges = 0

    if not os.path.isdir(directory_path):
        print(f"Error: Directory not found at '{directory_path}'")
        return 0, 0

    for filename in os.listdir(directory_path):
        print(filename)
        if filename.endswith(".json"):
            file_path = os.path.join(directory_path, filename)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                    # Count spatial edges within each frame
                    if "frame_states" in data and isinstance(data["frame_states"], dict):
                        for frame_key, frame_value in data["frame_states"].items():
                            if "edges" in frame_value and isinstance(frame_value["edges"], list):
                                total_spatial_edges = len(frame_value["edges"])


                    # Count temporal edges from the root
                    if "temporal_edges" in data and isinstance(data["temporal_edges"], list):
                        total_temporal_edges += len(data["temporal_edges"])

            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from {filename}. Skipping.")
            except Exception as e:
                print(f"An error occurred while processing {filename}: {e}")

    return total_spatial_edges, total_temporal_edges


# --- Execution ---
# NOTE: To run this code, create a folder named 'full_json' in the same directory
# as the script, and place your JSON files inside it.

# Define the directory containing your JSON files
json_directory =r"C:\Users\15972\Desktop\icra2026\results\full_scene_graph_json\full_scene_graph_json\office"

# Create a dummy directory and file for demonstration purposes
if not os.path.exists(json_directory):
    os.makedirs(json_directory)

# Call the function and get the counts
spatial_count, temporal_count = count_edge_types(json_directory)

# Print the final results
print(f"Total number of spatial edges: {spatial_count}")
print(f"Total number of temporal edges: {temporal_count}")