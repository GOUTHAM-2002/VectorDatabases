from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

# Sample product data with a larger dataset
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

# Create a list of product descriptions
descriptions = [product["description"] for product in products]

# Initialize TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(descriptions)

# Function to search for products based on a query
def search_products(query):
    # Transform the query using the same vectorizer
    query_vector = tfidf_vectorizer.transform([query])

    # Calculate similarities between the query and product embeddings
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # Create a list of products with their similarity scores
    sorted_products = sorted(zip([product["name"] for product in products], similarities), key=lambda item: item[1], reverse=True)

    return sorted_products

# Visualization function
def visualize_embeddings(query_embedding, results):
    all_embeddings = tfidf_matrix.toarray()
    all_product_names = [product["name"] for product in products]

    # Reduce dimensions to 2D using PCA
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(all_embeddings)

    # Create a figure
    plt.figure(figsize=(12, 8))

    # Calculate similarity scores for color mapping
    similarity_scores = [score for _, score in results]
    normed_scores = (similarity_scores - np.min(similarity_scores)) / (np.max(similarity_scores) - np.min(similarity_scores))

    # Scatter plot for product embeddings
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1],
                          c=normed_scores, cmap='viridis', s=100, alpha=0.7)

    # Plot the query embedding
    if query_embedding is not None:
        query_reduced = pca.transform(query_embedding.toarray())
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

# Interactive loop for searching products
def interactive_search():
    print("Welcome to the Product Search System!")
    print("Type 'exit' to quit the search.")

    while True:
        query = input("\nEnter your search query: ")
        if query.lower() == 'exit':
            print("Exiting the search system. Goodbye!")
            break

        results = search_products(query)

        print(f"\nSearch results for: '{query}'")
        for name, score in results:
            print(f"Product: {name}, Similarity Score: {score:.4f}")

        # Visualize the embeddings
        query_embedding = tfidf_vectorizer.transform([query])
        visualize_embeddings(query_embedding, results)

# Run the interactive search system
if __name__ == "__main__":
    interactive_search()
