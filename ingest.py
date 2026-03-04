import os
import sys
import shutil
import argparse
from typing import List
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# .envから設定を読み込み
load_dotenv()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAGチュートリアル: ドキュメントのベクターDB化")
    parser.add_argument(
        "--chunking",
        choices=["semantic", "fixed"],
        default="semantic",
        help="チャンキング手法 (default: semantic)"
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help="ChromaDBの保存先パス (default: 環境変数VECTOR_DB_PATHまたは./chroma_db)"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="既存のベクターDBを削除してから再構築する"
    )
    # Fixed Size Chunking のパラメータ
    parser.add_argument("--chunk-size", type=int, default=500, help="チャンクサイズ (fixed時のみ, default: 500)")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="チャンクオーバーラップ (fixed時のみ, default: 50)")
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    # DB パス: CLI引数 > 環境変数 > デフォルト
    persist_directory = args.db_path or os.getenv("VECTOR_DB_PATH", "./chroma_db")

    if args.reset:
        if os.path.exists(persist_directory):
            print(f"既存のベクターDBを削除しています: {persist_directory}")
            shutil.rmtree(persist_directory)
        else:
            print("削除対象のベクターDBが見当たりません。")

    # docsフォルダ内の全.txtファイルを自動探索
    docs_dir = "./docs"
    txt_files = sorted([
        os.path.join(docs_dir, f)
        for f in os.listdir(docs_dir)
        if f.endswith(".txt")
    ]) if os.path.isdir(docs_dir) else []

    all_documents = []
    for doc in txt_files:
        print(f"読み込み中: {doc}")
        loader = TextLoader(doc, encoding='utf-8')
        all_documents.extend(loader.load())

    if not all_documents:
        print("読み込むドキュメントが見つかりませんでした。")
        return

    # Embedding モデル（両手法で共通）
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-small"
    )

    # チャンキング手法の選択
    if args.chunking == "semantic":
        print("チャンキング手法: Semantic Chunking")
        # 埋め込みベクトルの類似度変化を見て、意味的な切れ目でテキストを分割する
        # breakpoint_threshold_type:
        #   "percentile"  : 類似度変化が上位95パーセンタイルの箇所で切る（デフォルト）
        #   "standard_deviation": 標準偏差を超える変化点で切る
        #   "interquartile" : 四分位範囲を超える変化点で切る
        text_splitter = SemanticChunker(
            embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95,
        )
    else:
        print(f"チャンキング手法: Fixed Size Chunking (size={args.chunk_size}, overlap={args.chunk_overlap})")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )

    chunks = text_splitter.split_documents(all_documents)
    print(f"チャンク数: {len(chunks)}")

    # チャンクの統計情報を表示
    if chunks:
        sizes = [len(c.page_content) for c in chunks]
        print(f"平均チャンクサイズ: {sum(sizes) / len(sizes):.0f} 文字")
        print(f"最小チャンクサイズ: {min(sizes)} 文字")
        print(f"最大チャンクサイズ: {max(sizes)} 文字")

    if os.path.exists(persist_directory) and not args.reset:
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