import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

def main() -> None:
    persist_directory = "./chroma_db"
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")
    
    vector_db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

    # LLMの設定
    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-latest",
        temperature=0,
    )

    # 1. 検索機 (Retriever) の設定
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    # 2. システムプロンプト（指示書き）の作成
    system_prompt = (
        "あなたは法務の専門家です。提供されたコンテキスト情報のみを使用して、質問に正確に答えてください。"
        "回答の際は、どのドキュメントの「第何条」に基づいた情報であるかを明示してください。"
        "コンテキストに情報が含まれていない場合は「その質問に関する情報は提供されたドキュメント内には見当たりません」と答えてください。"
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # 3. チェーンの構築 (LCELスタイルの新しい書き方)
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # 4. 実行
    print("\n--- RAGシステム起動 ---")
    print("質問を入力してください（'exit' または 'quit' で終了）")

    while True:
        query = input("\n質問 > ")
        if query.lower() in ["exit", "quit", "終了", "しゅうりょう"]:
            print("RAGシステムを終了します。")
            break
        
        if not query.strip():
            continue

        print("検索中...")
        try:
            response = rag_chain.invoke({"input": query})
        except Exception as e:
            print(f"エラーが発生しました: {e}")
            continue

        # 5. 結果の表示
        print("\n【回答】")
        print(response["answer"])
        
        print("\n【参照したソースの詳細】")
        sources = set()
        for doc in response["context"]:
            source = doc.metadata.get('source', '不明')
            sources.add(source)
        
        for i, source in enumerate(sources):
            print(f"Source {i+1}: {source}")

if __name__ == "__main__":
    main()