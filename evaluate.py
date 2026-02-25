import os
import re
import time
import json
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

def parse_test_set(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # セパレーターで分割
    sections = re.split(r'-{40,}', content)
    test_cases = []
    
    for section in sections:
        if "質問：" not in section:
            continue
            
        case = {}
        # IDとタイプのリマインダー
        match_id = re.search(r'(Q\d+)', section)
        if match_id:
            case['id'] = match_id.group(1)
            
        # 質問の抽出
        match_q = re.search(r'質問：(.+)', section)
        if match_q:
            case['question'] = match_q.group(1).strip()
            
        # 期待される正解の抽出
        match_a = re.search(r'期待される正解.*?：\s*(.+?)(?=\n\n|\n根拠文書：)', section, re.DOTALL)
        if match_a:
            case['expected_answer'] = match_a.group(1).strip()
            
        # 根拠文書の抽出
        match_e = re.search(r'根拠文書：(.+)', section)
        if match_e:
            case['expected_evidence'] = match_e.group(1).strip()
            
        if 'question' in case:
            test_cases.append(case)
            
    return test_cases

def get_rag_chain():
    persist_directory = "./chroma_db"
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")
    
    vector_db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)

    system_prompt = (
        "あなたは法務の専門家です。提供されたコンテキスト情報のみを使用して、質問に正確に答えてください。"
        "回答の際は、どのドキュメントの「第何条」に基づいた情報であるかを明示してください。"
        "コンテキストに情報が含まれていない場合は「その質問に関する情報は提供されたドキュメント内には見当たりません」と答えてください。"
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    return rag_chain

def evaluate_generation(llm, question, expected, generated):
    eval_prompt = f"""
以下のRAGシステムの回答を、期待される正解と比較して0-3点で採点してください。

質問: {question}
期待される正解: {expected}
生成された回答: {generated}

採点基準:
3点: すべての要素が含まれ、誤情報がない
2点: 主要要素は含まれるが一部欠落がある
1点: 部分的に正しいが誤情報または重大な欠落がある
0点: 誤情報が多い、または全く回答できていない

以下の形式で出力してください:
Score: [点数]
Reason: [理由]
"""
    response = llm.invoke(eval_prompt)
    content = response.content
    if isinstance(content, list):
        content = "".join([part.get("text", "") if isinstance(part, dict) else str(part) for part in content])
    
    score_match = re.search(r'Score:\s*(\d)', content)
    score = int(score_match.group(1)) if score_match else 0
    reason_match = re.search(r'Reason:\s*(.+)', content, re.DOTALL)
    reason = reason_match.group(1).strip() if reason_match else "No reason provided"
    
    return score, reason

def evaluate_retrieval(expected_evidence, retrieved_docs):
    # 文書名が含まれているか簡易チェック (DOC-001 -> doc_01)
    retrieved_sources = [doc.metadata.get('source', '') for doc in retrieved_docs]
    
    # 簡易マッピング
    doc_map = {
        "DOC-001": "doc_01",
        "DOC-002": "doc_02",
        "DOC-003": "doc_03",
        "DOC-004": "doc_04"
    }
    
    expected_files = []
    for doc_id, file_id in doc_map.items():
        if doc_id in expected_evidence:
            expected_files.append(file_id)
            
    if not expected_files:
        return 0, "Unknown evidence"

    found_count = 0
    for file_id in expected_files:
        if any(file_id in src for src in retrieved_sources):
            found_count += 1
            
    if found_count == len(expected_files):
        return 3, f"Found all expected sources: {expected_files}"
    elif found_count > 0:
        return 2, f"Found partial sources. Expected: {expected_files}, Got: {retrieved_sources}"
    else:
        return 0, f"No expected sources found. Expected: {expected_files}, Got: {retrieved_sources}"

def main():
    test_cases = parse_test_set("qa_test_set.txt")
    rag_chain = get_rag_chain()
    eval_llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)
    
    results = []
    print(f"--- 評価開始 (全{len(test_cases)}問) ---")
    
    for case in test_cases:
        print(f"Evaluating {case['id']}...")
        try:
            response = rag_chain.invoke({"input": case['question']})
            time.sleep(5)  # Rate limit protection
            
            gen_score, gen_reason = evaluate_generation(
                eval_llm, 
                case['question'], 
                case['expected_answer'], 
                response['answer']
            )
            time.sleep(5)  # Rate limit protection
            
            ret_score, ret_reason = evaluate_retrieval(
                case['expected_evidence'], 
                response['context']
            )
            
            res = {
                "id": case['id'],
                "gen_score": gen_score,
                "ret_score": ret_score,
                "gen_reason": gen_reason,
                "ret_reason": ret_reason
            }
            results.append(res)
            
            # 中間結果を保存
            with open("eval_results_temp.json", "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"Error evaluating {case['id']}: {e}")
            if "RESOURCE_EXHAUSTED" in str(e):
                print("Rate limit reached. Sleeping for 30s...")
                time.sleep(30)
                # リトライロジックは簡略化のため無し
            continue
        
    # 集計
    avg_gen = sum(r['gen_score'] for r in results) / len(results)
    avg_ret = sum(r['ret_score'] for r in results) / len(results)
    
    print("\n--- 評価結果サマリー ---")
    print(f"Generation平均スコア: {avg_gen:.2f} / 3.0")
    print(f"Retrieval平均スコア: {avg_ret:.2f} / 3.0")
    print("\n詳細:")
    for r in results:
        print(f"[{r['id']}] Ret: {r['ret_score']}, Gen: {r['gen_score']}")
        print(f"  Gen Reason: {r['gen_reason'].split('\n')[0]}...")

if __name__ == "__main__":
    main()
