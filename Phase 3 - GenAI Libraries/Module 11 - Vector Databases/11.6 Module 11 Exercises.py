# 11.6 Module 11 Exercises (Halaman 133-134)
# Multi-collection search helper
def search_multi_collections(client, collection_names: list[str], query_embedding: list[float], top_k: int = 5):
    all_results = []
    for name in collection_names:
        col = client.get_collection(name)
        res = col.query(query_embeddings=[query_embedding], n_results=top_k, include=["documents", "distances"])
        for doc, dist in zip(res["documents"][0], res["distances"][0]):
            all_results.append((doc, dist, name))
    all_results.sort(key=lambda x: x[1])
    return all_results[:top_k]

# Tenant Vector Store
class TenantVectorStore:
    def __init__(self, client):
        self.client = client

    def get_tenant_collection(self, tenant_id: str):
        return self.client.get_or_create_collection(f"tenant_{tenant_id}")

if __name__ == "__main__":
    print("Module 11 Exercises utilities initialized.")
