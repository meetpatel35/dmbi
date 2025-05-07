# import numpy as np
# import pandas as pd
# from graphviz import Digraph
# # Function to calculate entropy
# def entropy(y):
#   unique_classes, counts = np.unique(y, return_counts=True)
#   probs = counts / len(y)
#   return -np.sum(probs * np.log2(probs + 1e-9)) # Adding small value to avoid log(0)
# # Function to calculate information gain
# def information_gain(X_column, y):
#   total_entropy = entropy(y)
#   values, counts = np.unique(X_column, return_counts=True)
 
#   weighted_entropy = sum(
#     (counts[i] / len(y)) * entropy(y[X_column == values[i]]) for i
# in range(len(values))
#   )
 
#   return total_entropy - weighted_entropy
# # Function to find the best split
# def best_split(X, y):

#   best_feature = None
#   best_gain = -1
#   for feature in range(X.shape[1]):
#     gain = information_gain(X[:, feature], y)
#     if gain >= best_gain:
#       best_gain = gain
#       best_feature = feature
#   return best_feature
# # Class for tree nodes
# class DecisionTreeNode:
#   def __init__(self, feature=None, value=None, left=None, right=None,
# result=None):
#     self.feature = feature
#     self.value = value
#     self.left = left
#     self.right = right
#     self.result = result
# # Function to build the decision tree
# def build_tree(X, y, depth=0, max_depth=5):
#   if len(np.unique(y)) == 1:
#     return DecisionTreeNode(result=y[0])
 
#   if depth >= max_depth:
#     unique, counts = np.unique(y, return_counts=True)
#     return DecisionTreeNode(result=unique[np.argmax(counts)])
 
#   best_feature = best_split(X, y)
#   if best_feature is None:
#     unique, counts = np.unique(y, return_counts=True)
#     return DecisionTreeNode(result=unique[np.argmax(counts)])
 
#   values = np.unique(X[:, best_feature])
#   if len(values) == 1:
#     return DecisionTreeNode(result=y[0])
 
#   left_indices = X[:, best_feature] <= np.median(X[:, best_feature])
#   right_indices = ~left_indices
 
#   left_subtree = build_tree(X[left_indices], y[left_indices], depth +
# 1, max_depth)
#   right_subtree = build_tree(X[right_indices], y[right_indices],
# depth + 1, max_depth)
 
#   return DecisionTreeNode(feature=best_feature, value=np.median(X[:,
# best_feature]), left=left_subtree, right=right_subtree)

# # Function to visualize the decision tree
# def visualize_tree(node, feature_names, graph=None, node_id=0):
#   if graph is None:
#     graph = Digraph(format="png")
#     graph.attr(dpi="75", size="12,12") # Increase resolution and size
#   current_node = str(node_id)
#   if node.result is not None:
#     graph.node(current_node, f"Leaf: {node.result}", shape="box",
# style="filled", fillcolor="lightblue")
#   else:
#     feature_label = f"{feature_names[node.feature]} <= {round(node.value, 2)}"

#     graph.node(current_node, feature_label, shape="ellipse",
# style="filled", fillcolor="lightgray")
   
#     left_child_id = node_id + 1
#     graph, left_child_id = visualize_tree(node.left, feature_names,
# graph, left_child_id)
#     graph.edge(current_node, str(left_child_id - 1), label="Yes")
#     right_child_id = left_child_id + 1
#     graph, right_child_id = visualize_tree(node.right,
# feature_names, graph, right_child_id)
#     graph.edge(current_node, str(right_child_id - 1), label="No")
#   return graph, node_id + 2
# # Function to make predictions using the manual tree
# def predict_tree(node, x):
#   if node.result is not None:
#     return node.result
#   if x[node.feature] <= node.value:
#     return predict_tree(node.left, x)
#   else:
#     return predict_tree(node.right, x)
# # Load data
# df = pd.read_csv("heart_patient_data.csv") # Replace with actual file path
# # Convert categorical features to numerical values
# df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})
# df["Smoking"] = df["Smoking"].map({"No": 0, "Yes": 1})
# df["Diabetes"] = df["Diabetes"].map({"No": 0, "Yes": 1})
# df["Heart Patient"] = df["Heart Patient"].map({"No": 0, "Yes": 1})

# # Define X and y
# X = df.drop(columns=["Heart Patient"]).values
# y = df["Heart Patient"].values
# feature_names = df.drop(columns=["Heart Patient"]).columns.tolist() # Extract feature names
# # Split into training and testing sets
# np.random.seed(42)
# indices = np.random.permutation(len(X))
# train_indices = indices[:400]
# test_indices = indices[400:]
# X_train, X_test = X[train_indices], X[test_indices]
# y_train, y_test = y[train_indices], y[test_indices]
# # Build decision tree
# tree = build_tree(X_train, y_train, max_depth=3)
# # Generate the tree visualization with feature names
# graph, _ = visualize_tree(tree, feature_names)
# graph.render("decision_tree") # Saves as decision_tree.png
# # Make predictions
# y_train_pred = np.array([predict_tree(tree, x) for x in X_train])
# y_test_pred = np.array([predict_tree(tree, x) for x in X_test])
# # Calculate accuracy
# train_acc = np.mean(y_train_pred == y_train) * 100
# test_acc = np.mean(y_test_pred == y_test) * 100
# # Print results
# print(f"Training Accuracy: {train_acc:.2f}%")
# print(f"Testing Accuracy: {test_acc:.2f}%")
# # Display the tree (for Jupyter Notebook environments)
# from IPython.display import display
# display(graph)















#another one
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Modified Dataset
data = {
    "Order_Value": [400, 600, 550, 200, 300, 250, 700, 100, 350, 480, 150, 670],
    "Order_History_Count": [2, 10, 7, 5, 1, 9, 6, 3, 4, 12, 1, 8],
    "Image_Proof_Submitted": ['Yes', 'No', 'Yes', 'No', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'No', 'Yes'],
    "Refund_Requested": ['no', 'Yes', 'Yes', 'no', 'Yes', 'no', 'Yes', 'Yes', 'no', 'Yes', 'Yes', 'no'],
    "Is_Fraud": ['no', 'yes', 'no', 'yes', 'yes', 'no', 'yes', 'no', 'no', 'yes', 'no', 'yes']
}

df = pd.DataFrame(data)

# Label encode categorical features
le = LabelEncoder()
df['Image_Proof_Submitted'] = le.fit_transform(df['Image_Proof_Submitted'])
df['Refund_Requested'] = le.fit_transform(df['Refund_Requested'])
df['Is_Fraud'] = le.fit_transform(df['Is_Fraud'])  # yes = 1, no = 0

# Features and target
X = df[['Order_Value', 'Order_History_Count', 'Image_Proof_Submitted', 'Refund_Requested']]
y = df['Is_Fraud']

# Train the Decision Tree with larger depth
clf = DecisionTreeClassifier(criterion="entropy", max_depth=6, random_state=0)
clf.fit(X, y)

# Plot the tree
plt.figure(figsize=(18, 10))
plot_tree(clf, feature_names=X.columns, class_names=["No Fraud", "Fraud"], filled=True)
plt.title("Multi-Feature Decision Tree for Fraud Detection")
plt.show()



















# import numpy as np
# import pandas as pd
# from graphviz import Digraph
# from IPython.display import display

# # Dummy dataset (10 rows)
# data = {
#     "Age": [55, 43, 67, 50, 60, 35, 70, 45, 52, 48],
#     "Gender": ["Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female"],
#     "Blood Pressure": [140, 130, 150, 120, 145, 110, 160, 135, 138, 125],
#     "Cholesterol": [220, 180, 240, 200, 230, 170, 250, 190, 210, 185],
#     "Smoking": ["Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No"],
#     "Diabetes": ["Yes", "No", "Yes", "No", "No", "No", "Yes", "Yes", "No", "Yes"],
#     "Heart Patient": ["Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No"]
# }

# df = pd.DataFrame(data)

# # Encode categorical variables
# df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})
# df["Smoking"] = df["Smoking"].map({"No": 0, "Yes": 1})
# df["Diabetes"] = df["Diabetes"].map({"No": 0, "Yes": 1})
# df["Heart Patient"] = df["Heart Patient"].map({"No": 0, "Yes": 1})

# # Define X and y
# X = df.drop(columns=["Heart Patient"]).values
# y = df["Heart Patient"].values
# feature_names = df.drop(columns=["Heart Patient"]).columns.tolist()

# # Split into training and testing sets
# np.random.seed(42)
# indices = np.random.permutation(len(X))
# train_indices = indices[:7]
# test_indices = indices[7:]
# X_train, X_test = X[train_indices], X[test_indices]
# y_train, y_test = y[train_indices], y[test_indices]

# # --- Decision Tree Implementation ---

# def entropy(y):
#     unique_classes, counts = np.unique(y, return_counts=True)
#     probs = counts / len(y)
#     return -np.sum(probs * np.log2(probs + 1e-9))  # Avoid log(0)

# def information_gain(X_column, y):
#     total_entropy = entropy(y)
#     values, counts = np.unique(X_column, return_counts=True)
#     weighted_entropy = sum(
#         (counts[i] / len(y)) * entropy(y[X_column == values[i]]) for i in range(len(values))
#     )
#     return total_entropy - weighted_entropy

# def best_split(X, y):
#     best_feature = None
#     best_gain = -1
#     for feature in range(X.shape[1]):
#         gain = information_gain(X[:, feature], y)
#         if gain >= best_gain:
#             best_gain = gain
#             best_feature = feature
#     return best_feature

# class DecisionTreeNode:
#     def __init__(self, feature=None, value=None, left=None, right=None, result=None):
#         self.feature = feature
#         self.value = value
#         self.left = left
#         self.right = right
#         self.result = result

# def build_tree(X, y, depth=0, max_depth=3):
#     if len(np.unique(y)) == 1:
#         return DecisionTreeNode(result=y[0])
#     if depth >= max_depth:
#         unique, counts = np.unique(y, return_counts=True)
#         return DecisionTreeNode(result=unique[np.argmax(counts)])
#     best_feature = best_split(X, y)
#     if best_feature is None:
#         unique, counts = np.unique(y, return_counts=True)
#         return DecisionTreeNode(result=unique[np.argmax(counts)])
#     values = np.unique(X[:, best_feature])
#     if len(values) == 1:
#         return DecisionTreeNode(result=y[0])
#     split_value = np.median(X[:, best_feature])
#     left_indices = X[:, best_feature] <= split_value
#     right_indices = ~left_indices
#     left_subtree = build_tree(X[left_indices], y[left_indices], depth + 1, max_depth)
#     right_subtree = build_tree(X[right_indices], y[right_indices], depth + 1, max_depth)
#     return DecisionTreeNode(feature=best_feature, value=split_value, left=left_subtree, right=right_subtree)

# def visualize_tree(node, feature_names, graph=None, node_id=0):
#     if graph is None:
#         graph = Digraph(format="png")
#         graph.attr(dpi="75", size="12,12")
#     current_node = str(node_id)
#     if node.result is not None:
#         graph.node(current_node, f"Leaf: {node.result}", shape="box", style="filled", fillcolor="lightblue")
#     else:
#         feature_label = f"{feature_names[node.feature]} <= {round(node.value, 2)}"
#         graph.node(current_node, feature_label, shape="ellipse", style="filled", fillcolor="lightgray")
#         left_child_id = node_id + 1
#         graph, left_child_id = visualize_tree(node.left, feature_names, graph, left_child_id)
#         graph.edge(current_node, str(left_child_id - 1), label="Yes")
#         right_child_id = left_child_id + 1
#         graph, right_child_id = visualize_tree(node.right, feature_names, graph, right_child_id)
#         graph.edge(current_node, str(right_child_id - 1), label="No")
#     return graph, node_id + 2

# def predict_tree(node, x):
#     if node.result is not None:
#         return node.result
#     if x[node.feature] <= node.value:
#         return predict_tree(node.left, x)
#     else:
#         return predict_tree(node.right, x)

# # Build and visualize tree
# tree = build_tree(X_train, y_train, max_depth=3)
# graph, _ = visualize_tree(tree, feature_names)
# graph.render("decision_tree")  # Saves as decision_tree.png

# # Predictions
# y_train_pred = np.array([predict_tree(tree, x) for x in X_train])
# y_test_pred = np.array([predict_tree(tree, x) for x in X_test])

# # Accuracy
# train_acc = np.mean(y_train_pred == y_train) * 100
# test_acc = np.mean(y_test_pred == y_test) * 100

# print(f"Training Accuracy: {train_acc:.2f}%")
# print(f"Testing Accuracy: {test_acc:.2f}%")

# # Display tree in notebook (if supported)
# display(graph)












#alternative code



# import numpy as np
# import pandas as pd
# from graphviz import Digraph
# from IPython.display import display

# # Dummy dataset (10 rows)
# data = {
#     "Age": [55, 43, 67, 50, 60, 35, 70, 45, 52, 48],
#     "Gender": ["Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female"],
#     "Blood Pressure": [140, 130, 150, 120, 145, 110, 160, 135, 138, 125],
#     "Cholesterol": [220, 180, 240, 200, 230, 170, 250, 190, 210, 185],
#     "Smoking": ["Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No"],
#     "Diabetes": ["Yes", "No", "Yes", "No", "No", "No", "Yes", "Yes", "No", "Yes"],
#     "Heart Patient": ["Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No"]
# }

# df = pd.DataFrame(data)

# # Encode categorical variables
# df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})
# df["Smoking"] = df["Smoking"].map({"No": 0, "Yes": 1})
# df["Diabetes"] = df["Diabetes"].map({"No": 0, "Yes": 1})
# df["Heart Patient"] = df["Heart Patient"].map({"No": 0, "Yes": 1})

# # Define X and y
# X = df.drop(columns=["Heart Patient"]).values
# y = df["Heart Patient"].values
# feature_names = df.drop(columns=["Heart Patient"]).columns.tolist()

# # Split into training and testing sets
# np.random.seed(42)
# indices = np.random.permutation(len(X))
# train_indices = indices[:7]
# test_indices = indices[7:]
# X_train, X_test = X[train_indices], X[test_indices]
# y_train, y_test = y[train_indices], y[test_indices]

# # --- Decision Tree Implementation ---

# def entropy(y):
#     unique_classes, counts = np.unique(y, return_counts=True)
#     probs = counts / len(y)
#     return -np.sum(probs * np.log2(probs + 1e-9))  # Avoid log(0)

# def information_gain(X_column, y):
#     total_entropy = entropy(y)
#     values, counts = np.unique(X_column, return_counts=True)
#     weighted_entropy = sum(
#         (counts[i] / len(y)) * entropy(y[X_column == values[i]]) for i in range(len(values))
#     )
#     return total_entropy - weighted_entropy

# def best_split(X, y):
#     best_feature = None
#     best_gain = -1
#     for feature in range(X.shape[1]):
#         gain = information_gain(X[:, feature], y)
#         if gain >= best_gain:
#             best_gain = gain
#             best_feature = feature
#     return best_feature

# class DecisionTreeNode:
#     def __init__(self, feature=None, value=None, left=None, right=None, result=None):
#         self.feature = feature
#         self.value = value
#         self.left = left
#         self.right = right
#         self.result = result

# def build_tree(X, y, depth=0, max_depth=3):
#     if len(np.unique(y)) == 1:
#         return DecisionTreeNode(result=y[0])
#     if depth >= max_depth:
#         unique, counts = np.unique(y, return_counts=True)
#         return DecisionTreeNode(result=unique[np.argmax(counts)])
#     best_feature = best_split(X, y)
#     if best_feature is None:
#         unique, counts = np.unique(y, return_counts=True)
#         return DecisionTreeNode(result=unique[np.argmax(counts)])
#     values = np.unique(X[:, best_feature])
#     if len(values) == 1:
#         return DecisionTreeNode(result=y[0])
#     split_value = np.median(X[:, best_feature])
#     left_indices = X[:, best_feature] <= split_value
#     right_indices = ~left_indices
#     left_subtree = build_tree(X[left_indices], y[left_indices], depth + 1, max_depth)
#     right_subtree = build_tree(X[right_indices], y[right_indices], depth + 1, max_depth)
#     return DecisionTreeNode(feature=best_feature, value=split_value, left=left_subtree, right=right_subtree)


# import matplotlib.pyplot as plt

# def plot_tree(node, feature_names, depth=0, pos=(0, 0), x_offset=1.5, y_offset=1.5, ax=None, parent_pos=None, label=None):
#     if ax is None:
#         fig, ax = plt.subplots(figsize=(12, 6))
#         ax.axis("off")

#     if node.result is not None:
#         ax.text(pos[0], pos[1], f"Leaf: {node.result}", ha="center", va="center",
#                 bbox=dict(boxstyle="round", facecolor="lightblue"))
#     else:
#         feature_label = f"{feature_names[node.feature]} <= {round(node.value, 2)}"
#         ax.text(pos[0], pos[1], feature_label, ha="center", va="center",
#                 bbox=dict(boxstyle="round", facecolor="lightgray"))

#         # Compute child positions
#         left_pos = (pos[0] - x_offset / (depth + 1), pos[1] - y_offset)
#         right_pos = (pos[0] + x_offset / (depth + 1), pos[1] - y_offset)

#         # Draw lines to children
#         ax.plot([pos[0], left_pos[0]], [pos[1], left_pos[1]], 'k-')
#         ax.plot([pos[0], right_pos[0]], [pos[1], right_pos[1]], 'k-')

#         # Draw edge labels
#         ax.text((pos[0] + left_pos[0]) / 2, (pos[1] + left_pos[1]) / 2, "Yes", fontsize=9)
#         ax.text((pos[0] + right_pos[0]) / 2, (pos[1] + right_pos[1]) / 2, "No", fontsize=9)

#         # Recursively draw children
#         plot_tree(node.left, feature_names, depth + 1, left_pos, x_offset, y_offset, ax)
#         plot_tree(node.right, feature_names, depth + 1, right_pos, x_offset, y_offset, ax)

#     if parent_pos is None:
#         plt.show()

# def visualize_tree(node, feature_names, graph=None, node_id=0):
#     if graph is None:
#         graph = Digraph(format="png")
#         graph.attr(dpi="75", size="12,12")
#     current_node = str(node_id)
#     if node.result is not None:
#         graph.node(current_node, f"Leaf: {node.result}", shape="box", style="filled", fillcolor="lightblue")
#     else:
#         feature_label = f"{feature_names[node.feature]} <= {round(node.value, 2)}"
#         graph.node(current_node, feature_label, shape="ellipse", style="filled", fillcolor="lightgray")
#         left_child_id = node_id + 1
#         graph, left_child_id = visualize_tree(node.left, feature_names, graph, left_child_id)
#         graph.edge(current_node, str(left_child_id - 1), label="Yes")
#         right_child_id = left_child_id + 1
#         graph, right_child_id = visualize_tree(node.right, feature_names, graph, right_child_id)
#         graph.edge(current_node, str(right_child_id - 1), label="No")
#     return graph, node_id + 2

# def predict_tree(node, x):
#     if node.result is not None:
#         return node.result
#     if x[node.feature] <= node.value:
#         return predict_tree(node.left, x)
#     else:
#         return predict_tree(node.right, x)

# # Build and visualize tree
# tree = build_tree(X_train, y_train, max_depth=3)
# plot_tree(tree, feature_names)

# # Predictions
# y_train_pred = np.array([predict_tree(tree, x) for x in X_train])
# y_test_pred = np.array([predict_tree(tree, x) for x in X_test])

# # Accuracy
# train_acc = np.mean(y_train_pred == y_train) * 100
# test_acc = np.mean(y_test_pred == y_test) * 100

# print(f"Training Accuracy: {train_acc:.2f}%")
# print(f"Testing Accuracy: {test_acc:.2f}%")


