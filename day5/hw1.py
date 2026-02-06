from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import requests

# 1. 建立 Qdrant Collection 並連接
client = QdrantClient(url="http://localhost:6333")
COLLECTION_NAME = "hw1_collection"

# 嘗試建立 Collection
try:
    if not client.collection_exists(collection_name=COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=4096, distance=Distance.COSINE),
        )
        print(f"Collection '{COLLECTION_NAME}' created.")
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists.")
except Exception as e:
    print(f"Error checking/creating collection: {e}")

# 2. 建立五個 Point 或更多 (Raw Data)
raw_data = [
    {"id": 1, "text": "Python 是一種廣泛使用的高階程式語言，非常適合初學者。"},
    {"id": 2, "text": "Qdrant 是一個開源的高效向量資料庫，支援 Rust 編寫。"},
    {"id": 3, "text": "機器學習模型需要大量的數據進行訓練。"},
    {"id": 4, "text": "Docker 讓應用程式的部署變得更加輕量且一致。"},
    {"id": 5, "text": "VS Code 擁有多樣的擴充套件，能提升開發效率。"}
]

def get_embedding(text):
    """
    3. 使用 API 獲得向量
    """
    data = {
        "texts": [text],
        "normalize": True,
        "batch_size": 32
    }
    response = requests.post("https://ws-04.wade0426.me/embed", json=data)
    if response.status_code == 200:
        return response.json()['embeddings'][0]
    else:
        print(f"API Error: {response.status_code} - {response.text}")
        return None

# 4. 嵌入到 VDB
points_to_upsert = []
print("Generating embeddings and preparing points...")

for item in raw_data:
    vector = get_embedding(item["text"])
    if vector:
        point = PointStruct(
            id=item["id"],
            vector=vector,
            payload={"text": item["text"]}
        )
        points_to_upsert.append(point)

if points_to_upsert:
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points_to_upsert
    )
    print(f"Successfully upserted {len(points_to_upsert)} points to '{COLLECTION_NAME}'.")

# 5. 召回內容 (Retrieve/Search)
query_text = "好用的程式開發工具"
print(f"\nQuerying: {query_text}")
query_vector = get_embedding(query_text)

if query_vector:
    search_result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=3
    )

    print("\nSearch Results:")
    for point in search_result.points:
        print(f"ID: {point.id}")
        print(f"Score: {point.score:.4f}")
        print(f"Content: {point.payload['text']}")
        print("---")
