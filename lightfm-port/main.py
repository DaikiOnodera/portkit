import numpy as np
from lightfm.lightfm import LightFM
from lightfm.datasets import fetch_movielens
from lightfm.evaluation import precision_at_k

# Fix random seed for reproducible results
np.random.seed(42)

# Load the MovieLens 100k dataset. Only five
# star ratings are treated as positive.
data = fetch_movielens(min_rating=5.0)

# Instantiate and train the model with fixed random state
model = LightFM(loss='warp', random_state=42)
model.fit(data['train'], epochs=30, num_threads=2)

# Evaluate the trained model
test_precision = precision_at_k(model, data['test'], k=5).mean()

# Print results for comparison with PHP version
print(f"Test precision@5: {test_precision:.6f}")
print(f"Model parameters shape:")
print(f"  User embeddings: {model.user_embeddings.shape}")
print(f"  Item embeddings: {model.item_embeddings.shape}")
print(f"  User biases: {model.user_biases.shape}")
print(f"  Item biases: {model.item_biases.shape}")
