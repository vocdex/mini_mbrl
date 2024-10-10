
# Set the path to the output directory
output_dir="dataset"
echo "Generating dataset in $output_dir"
python data_generation.py --dir $output_dir
echo "Dataset generated in $output_dir"

# Visualize the dataset
echo "Creating visualization of the dataset"
python visualize_rollout.py --dir $output_dir --save_gif
echo "Visualization created"