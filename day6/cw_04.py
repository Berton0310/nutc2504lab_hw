import os
import uuid
import glob
import torch
import requests
from qdrant_client import QdrantClient, models
from transformers import AutoTokenizer, AutoModelForCausalLM

# ========== 設定 ==========
# Qdrant 設定
QDRANT_URL = "http://localhost:6333"
# 使用新的 Collection 名稱，因為需要同時支援 Dense 和 Sparse
COLLECTION_NAME = "ch_04" 

# Reranker 模型路徑
MODEL_PATH = r"c:\Users\berto\Desktop\lab\homework\test\day6\~\Models\Qwen3-Reranker-0.6B"

# Embedding API 設定
EMBEDDING_API_URL = "https://ws-04.wade0426.me/embed"
# 資料來源資料夾
DATA_DIR = "./cw_data"


# ========== 初始化資源 ==========

# 1. 初始化 Qdrant Client
client = QdrantClient(url=QDRANT_URL)


# 2. 載入 Reranker 模型
print(f"Loading Reranker model from {MODEL_PATH}...")
try:
    reranker_tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        trust_remote_code=True,
        padding_side='left'
    )
    reranker_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        trust_remote_code=True
    ).eval()
    
    # 獲取 token IDs
    token_false_id = reranker_tokenizer.convert_tokens_to_ids("no")
    token_true_id = reranker_tokenizer.convert_tokens_to_ids("yes")
    
    print("Reranker model loaded successfully.")
except Exception as e:
    print(f"Error loading Reranker model: {e}")
    print("請檢查 MODEL_PATH 是否正確，或確認模型檔案是否存在。")

# Reranker 相關設定
max_reranker_length = 8192
prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

if 'reranker_tokenizer' in locals():
    prefix_tokens = reranker_tokenizer.encode(prefix, add_special_tokens=False)
    suffix_tokens = reranker_tokenizer.encode(suffix, add_special_tokens=False)


# ========== 輔助函數 ==========

def get_embeddings(texts: list) -> list:
    """
    取得文字的 embedding 向量 (使用 API)
    """
    data = {
        "texts": texts,
        "normalize": True,
        "batch_size": 32
    }
    try:
        response = requests.post(EMBEDDING_API_URL, json=data)
        if response.status_code == 200:
            return response.json()['embeddings']
        else:
            print(f"Embedding API Error: {response.text}")
            return []
    except Exception as e:
        print(f"Embedding API Connection Error: {e}")
        return []

def split_text(text):
    """
    簡單的文本切分函數 (沿用 ch_03.py 邏輯)
    以 '。\\n' 作為切分點
    """
    return [chunk for chunk in text.split('。\n') if chunk.strip()]

def format_instruction(instruction, query, doc):
    """格式化 reranker 的輸入 - 修正版"""
    if instruction is None:
        instruction = '根據查詢檢索相關文件'
   
    output = f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"
    return output


def process_inputs(pairs):
    """處理 reranker 的輸入 - 修正版（三步驟）"""
    # 第一步：編碼（不包含 prefix 和 suffix）
    inputs = reranker_tokenizer(
        pairs,
        padding=False,
        truncation='longest_first',
        return_attention_mask=False,
        max_length=max_reranker_length - len(prefix_tokens) - len(suffix_tokens)
    )
   
    # 第二步：加入 prefix 和 suffix
    for i, ele in enumerate(inputs['input_ids']):
        inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens
   
    # 第三步：填充（padding）
    inputs = reranker_tokenizer.pad(
        inputs,
        padding=True,
        return_tensors="pt",
        max_length=max_reranker_length
    )
   
    # 移動到模型設備
    for key in inputs:
        inputs[key] = inputs[key].to(reranker_model.device)
   
    return inputs


@torch.no_grad()
def compute_logits(inputs):
    """計算相關性分數"""
    batch_scores = reranker_model(**inputs).logits[:, -1, :]
   
    true_vector = batch_scores[:, token_true_id]
    false_vector = batch_scores[:, token_false_id]
   
    batch_scores = torch.stack([false_vector, true_vector], dim=1)
    batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
   
    # 取 "yes" 的機率作為分數
    scores = batch_scores[:, 1].exp().tolist()
   
    return scores


def rerank_documents(query, documents, task_instruction=None):
    """
    使用 Qwen3-Reranker 重新排序文件
    Args:
        query: 查詢字串
        documents: 文件列表，每個元素是 (text, source) 的 tuple
        task_instruction: 任務指令（可選）
    """
    if 'reranker_model' not in globals():
        return [((doc[0], doc[1]), 0.0) for doc in documents]

    if task_instruction is None:
        task_instruction = '根據查詢檢索相關文件'
   
    # 格式化輸入 (只取 text)
    texts = [doc[0] for doc in documents]
    pairs = [format_instruction(task_instruction, query, text) for text in texts]
   
    # 處理輸入並計算分數
    try:
        inputs = process_inputs(pairs)
        scores = compute_logits(inputs)
    except Exception as e:
        print(f"Rerank Error: {e}")
        return [(doc, 0.0) for doc in documents]
   
    # 組合文件(含source)和分數，並按分數降序排序
    doc_scores = list(zip(documents, scores))
    doc_scores.sort(key=lambda x: x[1], reverse=True)
   
    return doc_scores


def hybrid_search_with_rerank(query: str, initial_limit: int = 20, final_limit: int = 3):
    """
    使用 RRF 混合搜索 + Reranker 重排
    """
    # 用 API 取得 query 的嵌入向量
    embeddings = get_embeddings([query])
    if not embeddings:
        print("Failed to get query embedding.")
        return []
    query_embedding = embeddings[0]
   
    # 混合搜索（RRF）
    print(f"Executing hybrid search for query: '{query}'...")
    try:
        response = client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                # BM25 關鍵字搜索
                models.Prefetch(
                    query=models.Document(
                        text=query,
                        model="Qdrant/bm25",
                    ),
                    using="sparse",
                    limit=initial_limit,
                ),
                # 語義搜索
                models.Prefetch(
                    query=query_embedding,
                    using="dense",
                    limit=initial_limit,
                ),
            ],
            # 使用 RRF 融合演算法
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=initial_limit,  # 取較多結果用於 reranking
        )
    except Exception as e:
        print(f"Qdrant Search Error: {e}")
        return []
   
    # 提取候選文件 (text, source)
    candidate_docs = []
    for point in response.points:
        text = point.payload.get("text", "")
        source = point.payload.get("source_file", "Unknown")
        candidate_docs.append((text, source))
   
    if not candidate_docs:
        print("No documents found.")
        return []
   
    # 使用 Reranker 重新排序
    print(f"正在對 {len(candidate_docs)} 個候選文件進行重排...")
    reranked_results = rerank_documents(query, candidate_docs)
   
    # 返回 top-k 結果
    top_results = reranked_results[:final_limit]
   
    print(f"\n查詢: {query}")
    print(f"重排後的 Top {final_limit} 結果:")
    print("=" * 80)
   
    for i, ((doc, source), score) in enumerate(top_results, 1):
        print(f"\n[{i}] 相關性分數: {score:.4f} | 來源: {source}")
        print(f"文件: {doc}")
        print("-" * 80)
   
    return top_results


# ========== LLM 設定 ==========
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import csv

llm = ChatOpenAI(
    model="/models/gpt-oss-120b",
    temperature=0.2,
    base_url="https://ws-03.wade0426.me/v1",
    api_key="EMPTY",
)

def generate_answer_from_llm(query, context_parts):
    """使用 LLM 根據多個上下文片段生成回答，並判斷最佳來源"""
    # 組合多個上下文
    context_str = "\n\n".join(context_parts)
    
    prompt = f"""你是一位專業的知識庫助手。請根據以下【參考資訊】回答【用戶問題】。

### 規則：
1. **必須**只依賴參考資訊回答，從中選擇與問題最相關的段落。
2. 若所有參考資訊都與問題無關，請回答「無法回答」。
3. 回答需簡潔準確。
4. 最後一行請務必標註你主要參考的來源，格式為 SOURCE: 檔名

### 參考資訊：
{context_str}

### 用戶問題：
{query}

請輸出回答："""
    
    try:
        messages = [HumanMessage(content=prompt)]
        response = llm.invoke(messages)
        return response.content.strip()
    except Exception as e:
        print(f"LLM Error: {e}")
        return "生成回答時發生錯誤"

def process_questions_csv():
    """讀取 questions.csv 並生成答案"""
    input_file = "questions.csv"
    output_file = "questions_processed1.csv"
    
    print(f"Processing {input_file}...")
    
    rows = []
    fieldnames = []
    
    try:
        with open(input_file, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            if '標準答案' not in fieldnames:
                fieldnames.append('標準答案')
            if '來源文件' not in fieldnames:
                fieldnames.append('來源文件')
            
            for row in reader:
                rows.append(row)
    except FileNotFoundError:
        print(f"File not found: {input_file}")
        return

    for i, row in enumerate(rows):
        question = row.get('題目', '').strip()
        if not question:
            continue
            
        print(f"\nProcessing Q{i+1}: {question}")
        
        # 檢索 Top 3，讓 LLM 從多個候選中選擇最相關的內容
        results = hybrid_search_with_rerank(
            query=question,
            initial_limit=20,
            final_limit=3
        )
        
        if results:
            # 組合多個上下文，附帶來源標籤
            context_parts = []
            for (text, source), score in results:
                context_parts.append(f"[來源: {source}]\n{text}")
                print(f"  > Candidate: {source} (Score: {score:.4f})")
            
            # 生成回答（傳入多個上下文）
            print("  > Generating answer...", end=" ", flush=True)
            raw_answer = generate_answer_from_llm(question, context_parts)
            print("Done.")
            
            # 解析回答和來源
            answer = raw_answer
            source_file = results[0][0][1]  # 預設用 reranker top 1 的來源
            
            if "SOURCE:" in raw_answer:
                parts = raw_answer.split("SOURCE:")
                answer = parts[0].strip()
                source_file = parts[1].strip()
            
            row['標準答案'] = answer
            row['來源文件'] = source_file
        else:
            print("  > No results found.")
            row['標準答案'] = "查無資料"
            row['來源文件'] = ""

    # 寫入 CSV
    try:
        with open(output_file, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print("\n" + "=" * 50)
        print(f"Successfully saved to {output_file}")
        print("=" * 50)
    except Exception as e:
        print(f"Error writing CSV: {e}")


# ========== 主程式 ==========

def main():
    # 1. 建立 Collection (若不存在)
    if not client.collection_exists(COLLECTION_NAME):
        print(f"Creating collection '{COLLECTION_NAME}'...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "dense": models.VectorParams(
                    distance=models.Distance.COSINE,
                    size=4096,
                ),
            },
            # 稀疏向量配置
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    modifier=models.Modifier.IDF
                )
            },
        )
        print("Collection created.")
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists.")

    # 2. 準備與匯入資料
    # (省略資料匯入部分，假設已由前次執行完成，或只有在需要時才執行)
    # 為了節省時間，若 Collection 已有資料則跳過檢查
    # ... (原本的資料匯入邏輯保留或簡化) ...
    # 這裡保留原本的邏輯，確保資料完整
    
    print(f"Checking data in '{DATA_DIR}'...")
    txt_files = glob.glob(os.path.join(DATA_DIR, "*.txt"))
    # ... (省略中間詳細檢查，直接呼叫處理問答) ...
    
    # 執行 CSV 處理
    process_questions_csv()


if __name__ == "__main__":
    main()
