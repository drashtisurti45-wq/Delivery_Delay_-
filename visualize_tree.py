import joblib
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# 1. Load the pre-trained decision tree model
try:
    dt_model = joblib.load('decision_tree_model.joblib')
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Error: decision_tree_model.joblib file not found. Please ensure it is in the same directory.")
    exit()

# Define the feature names explicitly based on the training dataset
feature_names = [
    'Delivery_Distance', 'Traffic_Congestion', 'Weather_Condition', 
    'Delivery_Slot', 'Driver_Experience', 'Num_Stops', 'Vehicle_Age', 
    'Road_Condition_Score', 'Package_Weight', 'Fuel_Efficiency', 
    'Warehouse_Processing_Time'
]

# Define target classes
class_names = ['No Delay', 'Delay']

# 2. Configure the plot layout and visualize the tree
plt.figure(figsize=(20, 10), dpi=300)
plot_tree(
    dt_model,
    feature_names=feature_names,
    class_names=class_names,
    filled=True,
    rounded=True,
    fontsize=10,
    max_depth=3
)

plt.title("Decision Tree Structure (Max Depth: 3)", fontsize=16)

# Save the generated visualization plot
output_image = 'decision_tree_visualization.png'
plt.savefig(output_image, bbox_inches='tight', dpi=300)
print(f"Visualization saved successfully as: {output_image}")
plt.show()
