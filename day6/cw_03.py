import requests
import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


def get_embedding(text):
    """取得文字的 embedding 向量"""
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


def split_text(text):
    """
    根據 '。\n' 進行文字切分
    
    Args:
        text: 要切分的文字
    
    Returns:
        切分後的文字列表
    """
    # 根據 。\n 進行切分
    chunks = text.split("。\n")
    
    # 過濾空白區塊，並在每個區塊結尾加回句號（除了最後一個如果原本沒有）
    result = []
    for i, chunk in enumerate(chunks):
        chunk = chunk.strip()
        if chunk:
            # 如果不是最後一個區塊，加回句號
            if i < len(chunks) - 1:
                chunk = chunk + "。"
            result.append(chunk)
    
    return result


# 連接到 Qdrant
client = QdrantClient(url="http://localhost:6333")

# Collection 名稱
collection_name = "ch_03"

# ========== 以下為建立 DB 的程式碼（已註解）==========
# 檢查 collection 是否已存在
# if client.collection_exists(collection_name):
#     print(f"Collection '{collection_name}' 已存在")
# else:
#     # 建立新的 collection（embedding 維度為 4096）
#     client.create_collection(
#         collection_name=collection_name,
#         vectors_config=VectorParams(size=4096, distance=Distance.COSINE)
#     )
#     print(f"成功建立 collection: {collection_name}")

# 顯示 collection 資訊
# collection_info = client.get_collection(collection_name)
# print(f"Collection 資訊: {collection_info}")
# ========== 以上為建立 DB 的程式碼（已註解）==========


# ========== 處理 cw_data 資料夾中的所有 txt 檔案（已註解）==========
# if __name__ == "__main__":
#     cw_data_folder = "./cw_data"
#     
#     # 取得所有 txt 檔案
#     txt_files = [f for f in os.listdir(cw_data_folder) if f.endswith(".txt")]
#     txt_files.sort()  # 排序確保順序一致
#     
#     print(f"找到 {len(txt_files)} 個 txt 檔案")
#     
#     # 用於存放所有要插入的 points
#     all_points = []
#     point_id = 0
#     
#     for txt_file in txt_files:
#         file_path = os.path.join(cw_data_folder, txt_file)
#         print(f"\n處理檔案: {txt_file}")
#         
#         # 讀取檔案內容
#         with open(file_path, "r", encoding="utf-8") as f:
#             text = f.read()
#         
#         # 切分文字
#         chunks = split_text(text)
#         print(f"  切分成 {len(chunks)} 個區塊")
#         
#         # 對每個區塊進行 embedding 並加入 points
#         for i, chunk in enumerate(chunks):
#             print(f"  處理區塊 {i+1}/{len(chunks)}...", end=" ")
#             
#             # 取得 embedding
#             embedding = get_embedding(chunk)
#             
#             if embedding:
#                 point = PointStruct(
#                     id=point_id,
#                     vector=embedding,
#                     payload={
#                         "source_file": txt_file,
#                         "chunk_index": i,
#                         "text": chunk
#                     }
#                 )
#                 all_points.append(point)
#                 point_id += 1
#                 print("完成")
#             else:
#                 print("失敗（無法取得 embedding）")
#     
#     # 批量插入到 Qdrant
#     print(f"\n正在將 {len(all_points)} 個向量插入到 Qdrant...")
#     
#     if all_points:
#         client.upsert(
#             collection_name=collection_name,
#             points=all_points
#         )
#         print(f"成功插入 {len(all_points)} 個向量到 collection '{collection_name}'")
#     else:
#         print("沒有資料可插入")
#     
#     # 顯示 collection 資訊
#     collection_info = client.get_collection(collection_name)
#     print(f"\nCollection 資訊: 共有 {collection_info.points_count} 個向量")
# ========== 以上為處理 cw_data 的程式碼（已註解）==========


# ========== Query Rewrite 與檢索功能 ==========
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

rewrite_prompt = """
# Role
你是一個 RAG (Retrieval-Augmented Generation) 系統的查詢重寫專家 (Query Rewriter)。
你的任務是將使用者的「最新問題」，結合「對話歷史」，重寫成一個適合讓向量資料庫或搜尋引擎理解的「獨立搜尋語句」。

# Context & Data Knowledge
我們的知識庫包含以下領域的資訊，請在重寫時參考這些背景：
1. **Google Cloud 硬體**：包含 Ironwood (第7代 TPU)、Axion 處理器、C4A (裸機)、N4A (虛擬機器)、GKE 支援等。
2. **AI 開發工具**：Google LiteRT、TensorFlow Lite、NPU/GPU 加速、裝置端推論。
3. **氣象與生活**：台中天氣預報、台灣氣候特徵 (季風/地形雨)、日本旅遊與流感疫情 (H3N2/B型)。

# Rules
1. **指代消解 (Coreference Resolution)**：將「它」、「那個」、「第二個」、「那邊」等代名詞，替換為對話歷史中提到的具體實體 (如：N4A, 台中, LiteRT)。
2. **補全上下文 (Contextualization)**：如果問題簡短（例如「效能如何？」），請補上主詞（例如「Google N4A 虛擬機器的效能如何？」）。
3. **保留原意**：不要回答問題，只要「重寫問題」。不要自行捏造不存在的資訊。
4. **關鍵字增強**：如果使用者的詞彙模糊，請嘗試加入上述 Knowledge 中的專有名詞（例如將「Google 新出的那個 CPU」重寫為包含 `Axion` 或 `Ironwood` 的語句），但不要過度發散。
5. **語言一致性**：輸出必須是繁體中文。

# Output Format
請直接輸出重寫後的搜尋語句，不要包含任何解釋或標點符號以外的文字，絕對禁止輸出任何思考過程、前言、解釋。
"""

# LLM 設定
llm = ChatOpenAI(
    base_url="https://ws-03.wade0426.me/v1",
    api_key="EMPTY",
    model="/models/gpt-oss-120b",
    temperature=0.7
)


def query_rewrite(user_query: str, chat_history: list = None) -> str:
    """
    使用 LLM 重寫使用者查詢
    
    Args:
        user_query: 使用者的原始問題
        chat_history: 對話歷史（可選）
    
    Returns:
        重寫後的查詢
    """
    # 建立對話歷史字串
    history_str = ""
    if chat_history:
        for msg in chat_history:
            role = "使用者" if msg["role"] == "user" else "助理"
            history_str += f"{role}: {msg['content']}\n"
    
    # 建立完整的使用者訊息
    if history_str:
        full_message = f"對話歷史：\n{history_str}\n最新問題：{user_query}"
    else:
        full_message = f"最新問題：{user_query}"
    
    # 呼叫 LLM
    messages = [
        SystemMessage(content=rewrite_prompt),
        HumanMessage(content=full_message)
    ]
    
    response = llm.invoke(messages)
    return response.content.strip()


def search_qdrant(query: str, top_k: int = 3) -> list:
    """
    在 Qdrant 中進行向量搜尋
    
    Args:
        query: 搜尋查詢
        top_k: 返回結果數量
    
    Returns:
        搜尋結果列表
    """
    # 取得查詢的 embedding
    query_embedding = get_embedding(query)
    
    if not query_embedding:
        print("無法取得查詢的 embedding")
        return []
    
    # 進行向量搜尋
    results = client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=top_k
    )
    
    return results


# 回答生成 prompt
answer_prompt = """你是一位專業的知識庫助手。請嚴格根據以下提供的【參考資訊】來回答用戶的問題。

### 規則：
1. **必須**只依賴【參考資訊】中的內容回答。
2. 如果【參考資訊】中沒有足夠的資訊來回答問題，請直接回答：「抱歉，根據目前的資料庫，我無法回答這個問題。」，**絕對不要**憑空捏造或使用你的外部知識。
3. 回答應簡潔、準確且專業。
4. **最後必須列出你參考的來源檔案名稱**，格式為 `SOURCE: filename1, filename2`。若無法回答則不需列出來源。

### 參考資訊：
{context_str}

### 用戶問題：
{query_str}

現在請開始輸出你的回答："""


def generate_answer(user_query: str, search_results: list) -> tuple[str, str]:
    """
    根據檢索結果生成回答
    
    Args:
        user_query: 使用者的原始問題
        search_results: Qdrant 檢索結果
    
    Returns:
        tuple[str, str]: (回答內容, 來源字串)
    """
    # 組合檢索結果為 context
    context_parts = []
    for i, result in enumerate(search_results):
        text = result.payload.get('text', '')
        source = result.payload.get('source_file', 'N/A')
        context_parts.append(f"[來源: {source}]\n{text}")
    
    context_str = "\n\n".join(context_parts)
    
    # 填入 prompt 模板（使用原始問題）
    filled_prompt = answer_prompt.format(
        context_str=context_str,
        query_str=user_query
    )
    
    # 呼叫 LLM 生成回答
    messages = [
        HumanMessage(content=filled_prompt)
    ]
    
    response = llm.invoke(messages)
    content = response.content.strip()
    
    # 解析回答和來源
    answer = content
    source_str = ""
    
    # 尋找 SOURCE: 標記
    if "SOURCE:" in content:
        parts = content.split("SOURCE:")
        answer = parts[0].strip()
        source_str = parts[1].strip()
    elif "抱歉，根據目前的資料庫，我無法回答這個問題" in content:
        # 如果無法回答，嘗試從檢索結果中填入來源（或保持空白，視需求而定）
        # 這裡選擇填入最相關的來源，以便追蹤
        if search_results:
             source_str = search_results[0].payload.get('source_file', '')
    
    return answer, source_str


# ========== 主程式：處理 CSV 檔案 ==========
import csv
from collections import defaultdict

if __name__ == "__main__":
    input_csv = "Re_Write_questions.csv"
    output_csv = "Re_Write_questions_processed_new.csv"
    
    # ... (省略讀取 CSV 部分，保持不變) ...
    # 讀取 CSV
    rows = []
    try:
        with open(input_csv, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            for row in reader:
                rows.append(row)
        print(f"成功讀取 {len(rows)} 筆資料")
    except FileNotFoundError:
        print(f"找不到檔案: {input_csv}")
        exit()

    # 確保有 answer 和 source 欄位
    if 'answer' not in fieldnames:
        fieldnames.append('answer')
    if 'source' not in fieldnames:
        fieldnames.append('source')
    
    # 用於儲存每個 conversation_id 的對話歷史
    conversations = defaultdict(list)
    
    print("=" * 50)
    print("開始處理 CSV...")
    print("=" * 50)
    
    for i, row in enumerate(rows):
        cid = row.get('conversation_id', 'default')
        qid = row.get('questions_id', str(i+1))
        question = row.get('questions', '').strip()
        
        if not question:
            continue
            
        print(f"\n[處理中] Conv ID: {cid}, Q ID: {qid}")
        print(f"原始問題: {question}")
        
        # 取得該對話的歷史紀錄
        chat_history = conversations[cid]
        
        # Step 1: Query Rewrite
        print("  Step 1: Rewrite...", end=" ", flush=True)
        rewritten_query = query_rewrite(question, chat_history)
        print("完成")
        
        # Step 2: 向量檢索 (改回 Top 3)
        print("  Step 2: Retrieval...", end=" ", flush=True)
        search_results = search_qdrant(rewritten_query, top_k=3)
        print(f"找到 {len(search_results)} 筆結果")
        
        # Step 3: 生成回答 (由 LLM 決定來源)
        print("  Step 3: Generating Answer...", end=" ", flush=True)
        answer, source_from_llm = generate_answer(question, search_results)
        print("完成")
        
        # 更新 row 資料
        row['answer'] = answer
        row['source'] = source_from_llm
        
        # 更新對話歷史
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": answer})
        
        conversations[cid] = chat_history

    # ... (省略寫入 CSV 部分，保持不變) ...
    # 寫入結果 CSV
    try:
        with open(output_csv, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print("\n" + "=" * 50)
        print(f"處理完成！結果已儲存至: {output_csv}")
        print("=" * 50)
    except Exception as e:
        print(f"寫入 CSV 失敗: {e}")