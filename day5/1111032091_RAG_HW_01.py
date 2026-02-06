from langchain_text_splitters import CharacterTextSplitter, TokenTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import requests
import uuid

# --- Helper Functions ---
def get_embedding(text):
    data = {
        "texts": [text],
        "normalize": True,
        "batch_size": 32
    }
    # 使用與 hw1 相同的 API
    try:
        response = requests.post("https://ws-04.wade0426.me/embed", json=data)
        if response.status_code == 200:
            return response.json()['embeddings'][0]
    except Exception as e:
        print(f"Embedding API Error: {e}")
    return None

client = QdrantClient(url="http://localhost:6333")

def setup_collection_and_upsert(collection_name, chunks, splitter_name, source_name):
    print(f"\nProcessing {collection_name} for {splitter_name} from {source_name}...")
    
    # 檢查/建立 Collection
    if not client.collection_exists(collection_name=collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=4096, distance=Distance.COSINE),
        )
        print(f"Collection '{collection_name}' created.")
    else:
        print(f"Collection '{collection_name}' already exists.")

    points = []
    for i, chunk_text in enumerate(chunks):
        # 為了避免 API 雖然切分了但內容為空
        if not chunk_text.strip():
            continue
            
        vector = get_embedding(chunk_text)
        if vector:
            points.append(PointStruct(
                id=str(uuid.uuid4()), # 使用 UUID 防止 ID 衝突
                vector=vector,
                payload={
                    "text": chunk_text,
                    "chunk_id": i,
                    "splitter": splitter_name,
                    "source": source_name
                }
            ))
            print(f"Generated embedding for chunk {i+1}/{len(chunks)}", end='\r')
    
    if points:
        client.upsert(
            collection_name=collection_name,
            points=points
        )
        print(f"\nSuccessfully upserted {len(points)} points to {collection_name}")
    else:
        print("\nNo points generated.")

# --- Splitting Functions ---

def character_split_text(text, chunk_size=200, chunk_overlap=0):
    print(f"--- CharacterTextSplitter (Size: {chunk_size}, Overlap: {chunk_overlap}) ---")
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator="",
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    print(f"總共產生 {len(chunks)} 個分塊\n")
    for i, chunk in enumerate(chunks, 1):
        print(f"=== 分塊 {i} ===")
        print(f"長度: {len(chunk)} 字符")
        print(f"內容: {chunk.strip()}")
        print()
    return chunks

def token_split_text(text, chunk_size=200, chunk_overlap=50):
    print(f"\n--- TokenTextSplitter (Size: {chunk_size}, Overlap: {chunk_overlap}) ---")
    text_splitter_token = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        model_name="gpt-4",
    )
    chunks_token = text_splitter_token.split_text(text)
    print(f"原始文本長度: {len(text)} tokens")
    print(f"分塊數量: {len(chunks_token)}")
    for i, chunk in enumerate(chunks_token):
        print(f"分塊 {i+1}:")
        print(f" 長度: {len(chunk)} tokens")
    return chunks_token

def semantic_split_text(text, min_chunk_size=100, max_chunk_size=200):
    print(f"\n--- Semantic Text Splitter (Min: {min_chunk_size}, Max: {max_chunk_size}) ---")
    # 根據截圖實作，需確保已安裝 semantic-text-splitter
    try:
        from semantic_text_splitter import TextSplitter
    except ImportError:
        print("Error: semantic-text-splitter not installed. Please run `pip install semantic-text-splitter`.")
        return []

    # 設定範圍：依照傳入參數
    splitter = TextSplitter((min_chunk_size, max_chunk_size))
    
    chunks = splitter.chunks(text)
    
    print(f"總共產生 {len(chunks)} 個分塊\n")
    for i, chunk in enumerate(chunks, 1):
        print(f"=== 分塊 {i} ===")
        print(f"長度: {len(chunk)} 字符")
        print(f"內容: {chunk.strip()}")
        print()
    return chunks

def get_dynamic_split_params(text, base_chunk_size=200):
    """
    根據文本大小動態調整切分參數
    
    Args:
        text: 輸入文本
        base_chunk_size: 基礎chunk大小，默認200
    
    Returns:
        dict: 包含三種切分方法的參數
    """
    text_length = len(text)
    
    # 根據文本長度調整參數
    if text_length < 500:
        # 短文本：使用較小的chunk
        character_chunk_size = 100
        token_chunk_size = 100
        token_overlap = 20
        semantic_min = 50
        semantic_max = 100
        
    elif text_length < 2000:
        # 中等文本：使用標準chunk
        character_chunk_size = 200
        token_chunk_size = 200
        token_overlap = 50
        semantic_min = 100
        semantic_max = 200
        
    elif text_length < 5000:
        # 較長文本：使用較大chunk
        character_chunk_size = 300
        token_chunk_size = 300
        token_overlap = 75
        semantic_min = 150
        semantic_max = 300
        
    else:
        # 長文本：使用大chunk
        character_chunk_size = 500
        token_chunk_size = 500
        token_overlap = 100
        semantic_min = 200
        semantic_max = 400
    
    return {
        'character_split': {
            'chunk_size': character_chunk_size
        },
        'token_split': {
            'chunk_size': token_chunk_size,
            'chunk_overlap': token_overlap
        },
        'semantic_split': {
            'min_chunk_size': semantic_min,
            'max_chunk_size': semantic_max
        }
    }
#查看各切分方式參數
# for i in range(1,6):
#     with open(f"data/data_0{i}.txt", "r", encoding="utf-8") as f:
#         text = f.read()
#     params = get_dynamic_split_params(text)
#     print(f"File {i}: {len(text)} characters")
#     print(f"Dynamic Params: {params}\n")

import os

# 定義要處理的檔案列表
data_dir = "data"
files = [f for f in os.listdir(data_dir) if f.startswith("data_") and f.endswith(".txt")]

print(f"Found files to process: {files}")

# --- 執行部分 (資料庫建立完成後可註解) ---
# for filename in files:
#     file_path = os.path.join(data_dir, filename)
#     print(f"\n{'='*30}")
#     print(f"Processing file: {filename}")
#     print(f"{'='*30}")
    
#     with open(file_path, "r", encoding="utf-8") as f:
#         text = f.read()

#     # 取得動態參數
#     params = get_dynamic_split_params(text)
#     print(f"Dynamic Params: {params}\n")

#     # 1. 執行 Character Split
#     chunks_char = character_split_text(
#         text, 
#         chunk_size=params['character_split']['chunk_size']
#     )
#     # 建立 Collection: hw_character_split
#     setup_collection_and_upsert("hw_character_split", chunks_char, "CharacterTextSplitter", filename)


#     # 2. 執行 Token Split
#     chunks_token = token_split_text(
#         text,
#         chunk_size=params['token_split']['chunk_size'],
#         chunk_overlap=params['token_split']['chunk_overlap']
#     )
#     # 建立 Collection: hw_token_split
#     setup_collection_and_upsert("hw_token_split", chunks_token, "TokenTextSplitter", filename)


#     # 3. 執行 Semantic Split
#     chunks_semantic = semantic_split_text(
#         text,
#         min_chunk_size=params['semantic_split']['min_chunk_size'],
#         max_chunk_size=params['semantic_split']['max_chunk_size']
#     )
#     # 建立 Collection: hw_semantic_split
#     setup_collection_and_upsert("hw_semantic_split", chunks_semantic, "SemanticTextSplitter", filename)


import pandas as pd
import uuid

def retrieve_and_export_answers():
    # --- 讀取問題並進行檢索 ---
    print(f"\n{'='*30}")
    print("Starting Retrieval and Exporting to CSV")
    print(f"{'='*30}")

    # 讀取 CSV
    df_questions = pd.read_csv("data/questions.csv")

    # 用來儲存結果的列表
    all_results = []

    # 逐一檢索
    collection_list = ["hw_character_split", "hw_token_split", "hw_semantic_split"]

    for collection_name in collection_list:
        print(f"Searching in collection: {collection_name}")
        for index, row in df_questions.iterrows():
            qid = row['q_id']
            question = row['questions']
            
            # 取得問題的 embedding
            query_vector = get_embedding(question)
            
            if query_vector:
                try:
                    # 搜尋 (只取 Top 1)
                    search_result = client.query_points(
                        collection_name=collection_name,
                        query=query_vector,
                        limit=1 
                    )
                    
                    if search_result.points:
                        point = search_result.points[0]
                        payload = point.payload
                        
                        # 準備寫入 CSV 的資料
                        result_data = {
                            "id": str(uuid.uuid4()),
                            "q_id": qid,
                            "method": collection_name,  # 方法: 對應 collection 名稱
                            "splitter": payload.get('splitter', 'Unknown'),
                            "score": round(point.score, 4),
                            "content": payload.get('text', ''),
                            "source": payload.get('source', 'Unknown')
                        }
                        all_results.append(result_data)
                        
                        # 顯示進度
                        print(f"  Q{qid} found with score: {result_data['score']}")
                    
                except Exception as e:
                    print(f"  Search failed for Q{qid} in {collection_name}: {e}")
            else:
                print(f"  Failed to generate embedding for Q{qid}")

    # 建立 DataFrame 並存檔
    if all_results:
        final_df = pd.DataFrame(all_results)
        # 確保欄位順序
        columns_order = ["id", "q_id", "method", "splitter", "score", "content", "source"]
        final_df = final_df[columns_order]
        
        final_df.to_csv("final_answer.csv", index=False, encoding="utf-8-sig")
        print(f"\nSuccessfully saved {len(final_df)} results to 'final_answer.csv'")
    else:
        print("\nNo results found to save.")

# 執行檢索與匯出
# retrieve_and_export_answers()

# --- Check Answer Functions ---
SERVER_URL = "https://hw-01.wade0426.me/submit_answer"

def submit_homework(q_id, student_answer):
    payload = {
        "q_id": int(q_id),
        "student_answer": student_answer
    }
    try:
        response = requests.post(SERVER_URL, json=payload)
        if response.status_code == 200:
            print(f"Q{q_id} Submitted: {response.text}")
            try:
                return response.json()
            except:
                return None
        else:
            print(f"Q{q_id} Failed: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Error submitting Q{q_id}: {e}")
        return None

def check_answers(csv_file="1111032091_RAG_HW_01.csv"):
    """
    讀取 final_answer.csv，將答案分成三個批次（固定切分、滑動切分、語意切分）提交評分，
    並列出各批次的成功數、失敗題號以及總成功數。
    """
    try:
        df = pd.read_csv(csv_file)
        
        # 定義三個批次
        batches = {
            "固定切分 (Fixed)": df.iloc[0:20],
            "滑動切分 (Sliding)": df.iloc[20:40],
            "語意切分 (Semantic)": df.iloc[40:60]
        }
        
        print(f"Reading {csv_file}, submitting answers in 3 batches...")
        
        grand_total_success = 0
        results_summary = {}

        for batch_name, batch_data in batches.items():
            print(f"\n--- Processing {batch_name} ---")
            batch_success = 0
            failed_ids = []
            
            for index, row in batch_data.iterrows():
                q_id = row['q_id']
                content = row['content']
                
                result = submit_homework(q_id, content)
                
                if result and result.get("message") == "評分成功":
                    batch_success += 1
                else:
                    failed_ids.append(q_id)
            
            results_summary[batch_name] = {
                "success": batch_success,
                "total": len(batch_data),
                "failed_ids": failed_ids
            }
            grand_total_success += batch_success

        # 列印最終報告
        print(f"\n{'='*30}")
        print("最終評分結果報告")
        print(f"{'='*30}")
        
        for batch_name, stats in results_summary.items():
            print(f"{batch_name}:")
            print(f"  成功: {stats['success']}/{stats['total']}")
            if stats['failed_ids']:
                print(f"  失敗題號: {stats['failed_ids']}")
            print("-" * 20)
            
        print(f"總計評分成功總數: {grand_total_success}")
        print(f"{'='*30}")
            
    except FileNotFoundError:
        print(f"Error: {csv_file} not found.")
    except KeyError as e:
        print(f"Error: Missing column in CSV - {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

# 執行檢查答案
# check_answers()
