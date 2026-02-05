import os
import json
from typing import TypedDict, Literal, List, Annotated
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from langchain_google_vertexai import ChatVertexAI
# 導入自定義工具
from search_searxng import search_searxng
from vlm_read_website import vlm_read_website

# --- 配置 ---
CACHE_FILE = "hw4_cache.json"

llm = ChatOpenAI(
    base_url="https://ws-05.huannago.com/v1",
    api_key="EMPTY",
    model="Qwen3-VL-8B-Instruct-BF16.gguf",
    temperature=0.7
)
# llm = ChatVertexAI(
#     model="gemini-2.5-pro",
#     project="gen-lang-client-0342191491",  # <--- 關鍵！這裡填對，錢就從抵免額出
#     location="us-central1",     # 建議選 us-central1 資源最豐富
#     temperature=0.7
# )
# --- 狀態定義 ---
class State(TypedDict):
    question: str
    answer: str
    source: str  # CACHE / LLM
    search_results: List[dict]  # SearXNG 搜尋結果
    vlm_content: str  # VLM 讀取的內容
    loop_count: int   # 防止無限迴圈
    reasoning: str    # 規劃器的思考過程
    current_query: str
    decision: Literal["sufficient", "insufficient"] # 決策結果

# --- 輔助函數 ---
def load_cache():
    if not os.path.exists(CACHE_FILE):
        return {}
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}

def save_cache(new_data):
    current_data = load_cache()
    current_data.update(new_data)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(current_data, f, ensure_ascii=False, indent=4)

# --- 節點 (Nodes) ---

def check_cache_node(state: State):
    """1. 檢查快取"""
    print(f"\n[系統] 正在檢查快取：{state['question']}")
    cache_data = load_cache()
    
    if state['question'] in cache_data:
        print("--- 命中快取 (Cache Hit) ---")
        return {
            "answer": cache_data[state['question']],
            "source": "CACHE",
            "loop_count": 0
        }
    else:
        print("--- 未命中快取 (Cache Miss) ---")
        return {
            "source": "LLM", 
            "loop_count": 0,
            "vlm_content": "",
            "search_results": []
        }

class PlannerDecision(BaseModel):
    reasoning: str = Field(description="分析目前資訊是否足夠回答問題")
    decision: Literal["sufficient", "insufficient"] = Field(description="決定是否回答或繼續搜尋")

def planner_node(state: State):
    """2. 規劃器 / 決策節點"""
    current_loop = state.get("loop_count", 0)
    print(f"\n[系統] 規劃器正在思考... (目前迴圈次數: {current_loop})")
    
    # 檢查迴圈限制
    if current_loop >= 3:
        print("--- 達到迴圈限制，強制回答 ---")
        return {"decision": "sufficient", "reasoning": "達到迴圈限制"}

    prompt = f"""
    使用者問題: {state['question']}
    
    目前收集的資訊:
    {state.get('vlm_content', '無')}
    
    搜尋結果:
    {str(state.get('search_results', []))[:500]}...
    
    請判斷目前收集的資訊是否足以準確回答使用者的問題。
    如果是，輸出 'sufficient'。
    如果否，輸出 'insufficient' 以執行更多搜尋/研究。
    """
    
    structured_llm = llm.with_structured_output(PlannerDecision)
    result = structured_llm.invoke(prompt)
    
    print(f"--- 決策: {result.decision} ({result.reasoning}) ---")
    return {"decision": result.decision, "reasoning": result.reasoning}

def query_gen_node(state: State):
    """3. 生成搜尋關鍵字"""
    print("\n[系統] 正在生成搜尋關鍵字...")
    
    try:
        prompt = f"根據問題 '{state['question']}'，生成一個具體的 Google 搜尋關鍵字以尋找答案。僅輸出關鍵字文字。"
        # 加入超時控制 (如果模型支援 timeout 參數，否則標準 invoke 可能不支援，這裡加 try-except 是核心)
        response = llm.invoke(prompt)
        query = response.content.strip().replace('"', '')
        print(f"--- 關鍵字: {query} ---")
        return {"current_query": query}
    except Exception as e:
        print(f"❌ 生成關鍵字失敗: {e}")
        return {"current_query": state['question']} 

def search_tool_node(state: State):
    """4. 執行搜尋"""
    query = state.get("current_query", state["question"])
    print(f"\n[系統] 正在搜尋: {query}")
    
    results = search_searxng(query, limit=3)
    return {"search_results": results}

def vlm_process_node(state: State):
    """5. 使用 VLM 讀取網站"""
    results = state.get("search_results", [])
    current_loop = state.get("loop_count", 0)
    
    if not results:
        print("--- 沒有搜尋結果可供讀取 ---")
        return {
            "vlm_content": "未找到搜尋結果。",
            "loop_count": current_loop + 1
        }
    
    # 選擇第一個有效的 URL
    target_url = results[0].get("url")
    title = results[0].get("title", "無標題")
    
    if not target_url:
        return {
            "vlm_content": "搜尋結果中沒有有效的 URL。",
            "loop_count": current_loop + 1
        }
        
    print(f"\n[系統] VLM 正在讀取: {target_url}")
    content = vlm_read_website(target_url, title)
    
    return {
        "vlm_content": content,
        "loop_count": current_loop + 1
    }

def final_answer_node(state: State):
    """6. 生成最終答案"""
    print("\n[系統] 正在生成最終答案...")
    
    prompt = f"""
    問題: {state['question']}
    
    已驗證資訊:
    {state.get('vlm_content', '無內容')}
    
    搜尋上下文:
    {state.get('search_results', [])}
    
    請提供一個全面且親切的回答給使用者。
    """
    
    response = llm.invoke(prompt)
    answer = response.content
    
    # 更新快取
    save_cache({state['question']: answer})
    
    return {"answer": answer}

# --- 邊 (Edges) ---

def route_check_cache(state: State):
    if state.get("source") == "CACHE":
        return "end"
    return "planner"

# --- 圖建構 (Graph Construction) ---
workflow = StateGraph(State) 

workflow.add_node("check_cache", check_cache_node)
workflow.add_node("planner", planner_node)
workflow.add_node("query_gen", query_gen_node)
workflow.add_node("search_tool", search_tool_node)
workflow.add_node("vlm_process", vlm_process_node)
workflow.add_node("final_answer", final_answer_node)

workflow.set_entry_point("check_cache")

workflow.add_conditional_edges(
    "check_cache",
    route_check_cache,
    {
        "end": END,
        "planner": "planner"
    }
)

workflow.add_conditional_edges(
    "planner",
    lambda x: x['decision'],
    {
        "sufficient": "final_answer",
        "insufficient": "query_gen"
    }
)

workflow.add_edge("query_gen", "search_tool")
workflow.add_edge("search_tool", "vlm_process")
workflow.add_edge("vlm_process", "planner") # 迴圈檢查是否足夠

workflow.add_edge("final_answer", END)

app = workflow.compile()
print(app.get_graph().draw_ascii())
# --- 執行 ---
if __name__ == "__main__":
    user_input = input("我是全能查證 AI 助手，請問有什麼想知道的嗎？").strip()
    if not user_input:
        user_input = "最近誰爬了101大樓"

    print("\n" + "="*50)
    print("🔍 開始查證流程")
    print("="*50)
    
    # 使用 invoke 取代 stream
    result = app.invoke(
        {"question": user_input, "loop_count": 0},
        config={"recursion_limit": 50}
    )
    
    # 輸出最終結果
    print("\n" + "="*50)
    print("📝 查證結果")
    print("="*50)
    print(f"來源: {result.get('source', 'LLM')}")
    print(f"迴圈次數: {result.get('loop_count', 0)}")
    
    print("\n[回答]")
    print(result.get("answer", "無法生成答案"))
    
    # 顯示參考來源
    results = result.get('search_results', [])
    if results:
        print("\n[參考來源]")
        for idx, res in enumerate(results, 1):
            title = res.get('title', '無標題')
            url = res.get('url', '無連結')
            print(f"{idx}. {title} ({url})")