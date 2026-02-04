import json
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END, add_messages
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Import the provided ASR tool
import hw_asr

# 1. 設定模型 (LLM)
llm = ChatOpenAI(
    base_url="https://ws-02.wade0426.me/v1",
    api_key="EMPTY",
    model="Llama-3.3-70B-Instruct-NVFP4",
    temperature=0,
    max_tokens=4096
)

# 2. 定義 State (狀態)
class AgentState(TypedDict):
    # transcript: 原始逐字稿 (SRT format)
    transcript: str
    # detailed_notes: 詳細的逐字稿 (時間軸+台詞)
    detailed_notes: str
    # summary: 重點摘要
    summary: str
    # final_output: 最終合併結果
    final_output: str

# 3. 定義節點 (Nodes)

def asr_node(state: AgentState):
    """
    ASR 節點: 呼叫 hw_asr.main() 取得轉錄結果
    """
    print("\n[Node] ASR Running...")
    # hw_asr.main() returns the SRT text string
    srt_text = hw_asr.main()
    return {"transcript": srt_text}

def minutes_taker_node(state: AgentState):
    """
    Minutes Taker 節點: 整理詳細逐字稿
    """
    print("\n[Node] Minutes Taker Running...")
    transcript = state["transcript"]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位專業的會議記錄員。"),
        ("user", "請將以下逐字稿(SRT格式)整理成詳細的記錄，需要按時間軸與對應台詞逐一列出:\n\n{transcript}")
    ])
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"transcript": transcript})
    
    return {"detailed_notes": result}

def summarizer_node(state: AgentState):
    """
    Summarizer 節點: 整理重點摘要
    """
    print("\n[Node] Summarizer Running...")
    transcript = state["transcript"]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位專業的重點分析師。"),
        ("user", "請閱讀以下逐字稿，整理出精簡的重點摘要:\n\n{transcript}")
    ])
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"transcript": transcript})
    
    return {"summary": result}

def writer_node(state: AgentState):
    """
    Writer 節點: 合併結果
    """
    print("\n[Node] Writer Running...")
    detailed_notes = state["detailed_notes"]
    summary = state["summary"]
    
    final_report = f"""
# 會議/Podcast 轉錄報告

## 1. 詳細逐字稿 (時間軸)
{detailed_notes}

---

## 2. 重點摘要
{summary}
    """
    return {"final_output": final_report}

# 4. 組裝 Graph
workflow = StateGraph(AgentState)

# 加入節點
workflow.add_node("asr", asr_node)
workflow.add_node("minutes_taker", minutes_taker_node)
workflow.add_node("summarizer", summarizer_node)
workflow.add_node("writer", writer_node)

# 設定入口
workflow.set_entry_point("asr")

# 設定邊 (Edges)
# ASR 完成後，同時進行 Minutes Taker 和 Summarizer (平行)
workflow.add_edge("asr", "minutes_taker")
workflow.add_edge("asr", "summarizer")

# 兩者都完成後，匯聚到 Writer
workflow.add_edge("minutes_taker", "writer")
workflow.add_edge("summarizer", "writer")

# Writer 完成後結束
workflow.add_edge("writer", END)

# 編譯
app = workflow.compile()
print(app.get_graph().draw_ascii())

# 5. 執行
if __name__ == "__main__":
    try:
        print("Starting Pipeline...")
        # 初始狀態為空，ASR 會自己去 fetch 資料
        result = app.invoke({"transcript": ""})
        
        print("\n\n" + "="*30)
        print("FINAL OUTPUT")
        print("="*30)
        print(result["final_output"])
        
    except Exception as e:
        print(f"Execution Error: {e}")

