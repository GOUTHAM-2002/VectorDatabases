from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

# Sample product data
products = [
    {"name": "Wireless Mouse", "description": "Wireless mouse for work and gaming, ergonomic design."},
    {"name": "Bluetooth Headphones", "description": "Bluetooth headphones with great sound quality and noise cancellation."},
    {"name": "Laptop Stand", "description": "Adjustable stand for laptop use, improves posture and ergonomics."},
    {"name": "Smartphone Case", "description": "Stylish case for smartphone, shockproof and lightweight."},
    {"name": "LED Desk Lamp", "description": "LED lamp for desk use, with adjustable brightness and color temperature."},
    {"name": "Mechanical Keyboard", "description": "Gaming mechanical keyboard with RGB lighting and programmable keys."},
    {"name": "4K Monitor", "description": "Ultra HD 4K monitor, perfect for gaming and design work."},
    {"name": "Portable Charger", "description": "High-capacity portable charger for smartphones and tablets."},
    {"name": "Wireless Earbuds", "description": "True wireless earbuds with deep bass and long battery life."},
    {"name": "USB-C Hub", "description": "Multi-port USB-C hub for connecting multiple devices."},
    {"name": "Smartwatch", "description": "Feature-rich smartwatch with fitness tracking and notifications."},
    {"name": "External Hard Drive", "description": "1TB external hard drive for data storage and backups."},
    {"name": "Gaming Mouse Pad", "description": "Large gaming mouse pad with a smooth surface for precision."},
    {"name": "HDMI Cable", "description": "High-speed HDMI cable for connecting devices to monitors."},
    {"name": "Bluetooth Speaker", "description": "Portable Bluetooth speaker with high-quality sound and waterproof design."},
    {"name": "VR Headset", "description": "Virtual reality headset for immersive gaming experiences."},
    {"name": "Wireless Printer", "description": "All-in-one wireless printer for home and office use."},
    {"name": "Laptop Backpack", "description": "Spacious laptop backpack with multiple compartments for organization."},
    {"name": "Fitness Tracker", "description": "Wearable fitness tracker with heart rate monitoring and step counting."},
    {"name": "Camera Tripod", "description": "Adjustable camera tripod for stable shots and photography."},
    {"name": "Bluetooth Key Finder", "description": "Compact Bluetooth tracker for finding lost items easily."}
]

# Step 1: Preprocessing the product descriptions
sentences = [product["description"].lower().split() for product in products]

# Step 2: Training the Word2Vec model
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# Step 3: Function to get average Word2Vec embedding for a description
def get_average_embedding(description):
    words = description.lower().split()
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(model.vector_size)

# Step 4: Calculate embeddings for product descriptions
embeddings = {}
for product in products:
    description = product["description"]
    embeddings[product["name"]] = get_average_embedding(description)

# Step 5: Function to search for products based on a query
def search_products(query):
    query_embedding = get_average_embedding(query)
    similarities = {}
    for name, embedding in embeddings.items():
        similarity_score = cosine_similarity([query_embedding], [embedding])[0][0]
        similarities[name] = similarity_score
    sorted_products = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
    return sorted_products, query_embedding

# Step 6: Enhanced visualization using Matplotlib
def visualize_embeddings(query_embedding, results):
    all_embeddings = np.array(list(embeddings.values()))
    all_product_names = list(embeddings.keys())

    # Reduce dimensions to 2D using PCA for better visualization
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(all_embeddings)

    # Create a figure
    plt.figure(figsize=(12, 8))

    # Calculate similarity scores for color mapping
    similarity_scores = [score for _, score in results]
    normed_scores = (similarity_scores - np.min(similarity_scores)) / (
                np.max(similarity_scores) - np.min(similarity_scores))

    # Scatter plot for product embeddings
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1],
                          c=normed_scores, cmap='viridis', s=100, alpha=0.7)

    # Plot the query embedding
    if query_embedding is not None:
        query_reduced = pca.transform([query_embedding])
        plt.scatter(query_reduced[0][0], query_reduced[0][1], color='red', s=200, edgecolor='k', label='Query')

    # Annotate product names
    for i, name in enumerate(all_product_names):
        plt.annotate(name, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]), fontsize=9, ha='right')

    # Title and labels
    plt.title('2D Visualization of Product Embeddings with Query', fontsize=16)
    plt.xlabel('PCA Component 1', fontsize=12)
    plt.ylabel('PCA Component 2', fontsize=12)

    # Colorbar to indicate similarity scores
    cbar = plt.colorbar(scatter)
    cbar.set_label('Normalized Similarity Score', fontsize=12)

    # Show legend
    plt.legend(loc='upper right')

    # Show plot
    plt.grid()
    plt.tight_layout()
    plt.show()

# Step 7: Interactive loop for searching products
def interactive_search():
    print("Welcome to the Product Search System using Word Embeddings!")
    print("Type 'exit' to quit the search.")

    while True:
        query = input("\nEnter your search query: ")
        if query.lower() == 'exit':
            print("Exiting the search system. Goodbye!")
            break

        results, query_embedding = search_products(query)

        print(f"\nSearch results for: '{query}'")
        for name, score in results:
            print(f"Product: {name}, Similarity Score: {score:.4f}")

        # Visualize the embeddings
        visualize_embeddings(query_embedding, results)

# Run the interactive search system
if __name__ == "__main__":
    interactive_search()

