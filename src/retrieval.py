import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATA_PATH = r"C:\Users\ADMIN\OneDrive\Desktop\hiver-sde\data\processed\uber_clean.csv"


class HistoricalRetriever:

    def __init__(self, data_path=DATA_PATH):

        print("Loading historical Uber conversations...")

        self.df = pd.read_csv(data_path)

        self.df["customer_message"] = (
            self.df["customer_message"]
            .fillna("")
            .astype(str)
        )

        self.df["uber_response"] = (
            self.df["uber_response"]
            .fillna("")
            .astype(str)
        )

        print("Historical conversations:", len(self.df))

        # Create TF-IDF representation
        print("Building TF-IDF index...")

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_features=100000,
            sublinear_tf=True
        )

        self.matrix = self.vectorizer.fit_transform(
            self.df["customer_message"]
        )

        print("Retrieval index ready!")


    def retrieve(self, query, top_k=3):

        # Convert new customer message to TF-IDF
        query_vector = self.vectorizer.transform([query])

        # Calculate similarity
        similarities = cosine_similarity(
            query_vector,
            self.matrix
        ).flatten()

        # Get top results
        top_indices = similarities.argsort()[-top_k:][::-1]

        results = []

        for index in top_indices:

            results.append({
                "customer_message": self.df.iloc[index]["customer_message"],
                "uber_response": self.df.iloc[index]["uber_response"],
                "similarity": float(similarities[index])
            })

        return results


# ---------------------------------------------------------
# Test retrieval
# ---------------------------------------------------------

if __name__ == "__main__":

    retriever = HistoricalRetriever()

    test_messages = [
        "I was charged a cancellation fee even though I did not cancel my ride.",
        "My Uber app is not working.",
        "I lost my phone in an Uber.",
        "My driver was rude and behaved badly.",
        "How do I add a credit card to my account?"
    ]

    for message in test_messages:

        print("\n" + "=" * 80)
        print("CUSTOMER QUERY:")
        print(message)

        results = retriever.retrieve(message, top_k=3)

        print("\nTOP HISTORICAL MATCHES:")

        for i, result in enumerate(results, 1):

            print("\n--- Match", i, "---")
            print("Similarity:", round(result["similarity"], 4))

            print("\nHistorical customer:")
            print(result["customer_message"])

            print("\nHistorical Uber response:")
            print(result["uber_response"])