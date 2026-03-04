import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


def main() -> None:
    persist_directory = "./chroma_db"
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")
    
    vector_db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

    # 1. 検索機 (Retriever) の設定: ベクトル検索
    print("検索インデックスを準備中...")
    
    vector_retriever = vector_db.as_retriever(search_kwargs={"k": 10}) # リランク用に多めに取得

    # シンプルなベクトル検索を使用
    retriever = vector_retriever

    # LLMの設定
    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-latest",
        temperature=0,
    )

    # システムプロンプトの調整（より厳密な抽出を指示）
    system_prompt = (
        "あなたは法務・コンプライアンスの専門家です。提供されたコンテキスト情報（法律・規約文書）に基づいて、質問に正確に答えてください。"
        "回答のルール:\n"
        "1. 提供されたコンテキストに記載されている事実のみを述べてください。推測や一般的な知識で補完しないでください。\n"
        "2. 会社名、メールアドレス、数値（年率、日数、金額など）は、コンテキストにある通りに正確に引用してください。\n"
        "3. どの文書の「第何条」またはどの項目に基づいた情報であるかを明示してください。\n"
        "4. コンテキストに情報が含まれていない場合は「提供されたドキュメント内には見当たらないため、回答できません」と明言してください。\n"
        "\n\n"
        "コンテキスト:\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # チェーンの構築 (LCEL方式)
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("\n--- 改良RAGシステム (v2) 起動 ---")
    print("改良されたベクトル検索が有効です。")
    print("質問を入力してください（'exit' で終了）")

    while True:
        query = input("\n質問 > ")
        if query.lower() in ["exit", "quit", "終了"]:
            break
        
        if not query.strip():
            continue

        print("高度な検索と推論を実行中...")
        try:
            # リトリーバーでドキュメントを取得
            retrieved_docs = retriever.invoke(query)
            response = rag_chain.invoke(query)
        except Exception as e:
            print(f"エラーが発生しました: {e}")
            continue

        print("\n【回答】")
        print(response)
        
        print("\n【参照したソース (リランク済みTop-N)】")
        for i, doc in enumerate(retrieved_docs):
            source = doc.metadata.get('source', '不明')
            print(f"[{i+1}] {source}")

if __name__ == "__main__":
    main()
