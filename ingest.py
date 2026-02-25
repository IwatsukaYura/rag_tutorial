import os
import sys
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_experimental.text_splitter import SemanticChunker
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

    # 2. ベクトル化 (Embedding) ← Semantic Chunkerが内部で使用するため先に定義
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-small"
    )

    # 3. テキストの分割 (Semantic Chunking)
    # 埋め込みベクトルの類似度変化を見て、意味的な切れ目でテキストを分割する
    # breakpoint_threshold_type:
    #   "percentile"  : 類似度変化が上位95パーセンタイルの箇所で切る（デフォルト）
    #   "standard_deviation": 標準偏差を超える変化点で切る
    #   "interquartile" : 四分位範囲を超える変化点で切る
    text_splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95,  # 類似度変化の上位5%を分割点とする
    )
    chunks = text_splitter.split_documents(all_documents)
    print(f"セマンティック分割後のチャンク数: {len(chunks)}")


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