import os
import sys
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# .envから設定を読み込み
load_dotenv()

def main():
    persist_directory = "./chroma_db"

    # コマンドライン引数の処理
    if "--reset" in sys.argv:
        if os.path.exists(persist_directory):
            print(f"既存のベクターDBを削除しています: {persist_directory}")
            shutil.rmtree(persist_directory)
        else:
            print("削除対象のベクターDBが見当たりません。")

    # 1. ドキュメントの読み込み
    docs = [
        "doc_01_privacy_policy.txt",
        "doc_02_terms_of_service.txt",
        "doc_03_security_policy.txt",
        "doc_04_business_contract.txt"
    ]
    
    all_documents = []
    for doc in docs:
        if os.path.exists(doc):
            loader = TextLoader(doc, encoding='utf-8')
            all_documents.extend(loader.load())
    
    if not all_documents:
        print("読み込むドキュメントが見つかりませんでした。")
        return

    # 2. テキストの分割 (Chunking)
    # 条文を壊さないよう、少し大きめのサイズでオーバーラップを持たせる
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", "。"]
    )
    chunks = text_splitter.split_documents(all_documents)
    print(f"分割後のチャンク数: {len(chunks)}")

    # 3. ベクトル化 (Embedding)
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-small"
    )

    # 4. ベクターDB (Chroma) に保存
    # --reset が指定されていた場合は新規作成、そうでなければ既存に追加
    if os.path.exists(persist_directory) and "--reset" not in sys.argv:
        print("既存のベクターDBにドキュメントを追加しています...")
        vector_db = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        vector_db.add_documents(chunks)
    else:
        print("新規にベクターDBを作成しています...")
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_directory
        )
    
    print(f"ベクターDBの更新が完了しました。保存先: {persist_directory}")

if __name__ == "__main__":
    main()