import json
import os
from typing import List, Dict
from pathlib import Path

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.llms.base import LLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Transformers for your LoRA model
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig
from peft import PeftModel
from typing import Optional, List as ListType


class DocumentChunker:
    """Chunk documents and save one file per document"""

    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 50, output_dir: str = "chunked_docs"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Initialize LangChain text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size * 4,  # Approximate characters (4 chars per word)
            chunk_overlap=chunk_overlap * 4,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        print(f"DocumentChunker initialized - Output: {self.output_dir}")

    def chunk_and_save_one_document(self, doc):
        """
        Chunk documents and save one file per document

        Args:
          doc: document dictionaries

        """
        doc_id = doc.get('id', 'unknown_doc')
        doc_title = doc.get('title', 'Untitled')
        doc_content = doc.get('content', '')

        print(f"  Processing: {doc_title} ({len(doc_content.split())} words)")

        # Create LangChain document
        langchain_doc = Document(
            page_content=doc_content,
            metadata={
                'id': doc_id,
                'title': doc_title,
                'source': doc.get('source', 'unknown'),
                'url': doc.get('url', ''),
                'word_count': len(doc_content.split())
            }
        )

        # Split into chunks
        chunks = self.text_splitter.split_documents([langchain_doc])

        # Convert chunks to our format
        chunk_list = []
        for i, chunk in enumerate(chunks):
            chunk_dict = {
                'id': f"{doc_id}_chunk_{i}",
                'content': chunk.page_content,
                'chunk_index': i,
                'word_count': len(chunk.page_content.split()),
                'parent_document_id': doc_id,
                'parent_document_title': doc_title,
                'metadata': chunk.metadata
            }
            chunk_list.append(chunk_dict)

        # Save chunks for this document
        safe_doc_id = doc_id.replace('/', '_').replace('\\', '_').replace(':', '_')
        filename = f"{safe_doc_id}_chunks.json"
        file_path = self.output_dir / filename

        document_data = {
            'document_info': {
                'id': doc_id,
                'title': doc_title,
                'original_word_count': len(doc_content.split()),
                'chunk_count': len(chunk_list),
                'source': doc.get('source', 'unknown')
            },
            'chunks': chunk_list
        }

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(document_data, f, indent=2, ensure_ascii=False)

        print(f" ✅✅✅ Saved {len(chunk_list)} chunks to {filename}")

    def chunk_and_save_documents(self, guidelines_dir: str):
        """Load guidelines from local JSON files, chunk and save"""
        json_dir = Path(guidelines_dir) / "json"

        if not json_dir.exists():
            print(f"Guidelines directory not found: {json_dir}")
            print("Please run the guidelines collector first.")
            return False

        json_files = list(json_dir.glob("*.json"))
        if not json_files:
            print(f"No JSON files found in: {json_dir}")
            return False

        print(f"📚 Loading {len(json_files)} guidelines from local files...")

        loaded_count = 0
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    guideline_data = json.load(f)

                # Validate required fields
                if all(key in guideline_data for key in ['id', 'title', 'content']):
                    # self.guidelines.append(guideline_data)
                    self.chunk_and_save_one_document(guideline_data)
                    loaded_count += 1
                else:
                    print(f"⚠️ Skipping invalid file: {json_file.name}")

            except Exception as e:
                print(f"❌ Error loading {json_file.name}: {e}")

        print(f"📊 Chunking complete: {loaded_count} documents processed")


class CustomT5LLM(LLM):
    """Custom LLM wrapper for LoRA-tuned FLAN-T5"""

    def __init__(self, model_path: str):
        super().__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-base')
        peft_model_base = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base", torch_dtype=torch.bfloat16)
        self._model = PeftModel.from_pretrained(peft_model_base, model_path, torch_dtype=torch.bfloat16,
                                               is_trainable=False).to(device)
        print(f"✅ Loaded LoRA FLAN-T5 model from {model_path}")

    def _call(self, prompt: str, stop: Optional[ListType[str]] = None) -> str:
        device = next(self._model.parameters()).device

        ge_config_v2 = GenerationConfig(
            max_new_tokens=200,
            num_beams=3,
            early_stopping=True,
            repetition_penalty=1.2,  # Prevents repetition!
            no_repeat_ngram_size=3,  # No 3-gram repetition
            do_sample=False,
            pad_token_id=self._tokenizer.pad_token_id,
            eos_token_id=self._tokenizer.eos_token_id
        )

        # Generate answer
        inputs = self._tokenizer(prompt, return_tensors="pt").to(device)
        response = self._tokenizer.decode(
            self._model.generate(input_ids=inputs.input_ids, generation_config=ge_config_v2)[0],
            skip_special_tokens=True)
        return response.strip()

    @property
    def _llm_type(self) -> str:
        return "custom_t5"

class LangChainRAGSystem:
    """RAG system using LangChain with chunked documents"""

    def __init__(self, model_path: str, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize LangChain RAG system"""

        # Initialize your custom LLM
        self.llm = CustomT5LLM(model_path)

        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        )

        # Initialize components
        self.vectorstore = None
        self.retrieval_qa = None
        self.documents = []

        # Custom prompt template for medical QA
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""Based on the following medical information, provide a clear and accurate answer to the question.

Medical Information:
{context}

Question: {question}

Answer:"""
        )

        print("✅ LangChain RAG system initialized")

    def load_chunked_documents(self, chunked_docs_dir: str) -> None:
        """
        Load all chunked documents from directory

        Args:
            chunked_docs_dir: Directory containing document chunk files
        """
        chunks_path = Path(chunked_docs_dir)

        if not chunks_path.exists():
            raise FileNotFoundError(f"Chunked documents directory not found: {chunks_path}")

        # Find all chunk files
        chunk_files = list(chunks_path.glob("*_chunks.json"))

        if not chunk_files:
            raise FileNotFoundError(f"No chunk files found in {chunks_path}")

        print(f"📚 Loading chunks from {len(chunk_files)} files...")

        all_documents = []

        for chunk_file in chunk_files:
            try:
                with open(chunk_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                doc_info = data.get('document_info', {})
                chunks = data.get('chunks', [])

                print(f"  📄 {doc_info.get('title', 'Unknown')}: {len(chunks)} chunks")

                # Convert chunks to LangChain documents
                for chunk in chunks:
                    langchain_doc = Document(
                        page_content=chunk['content'],
                        metadata={
                            'chunk_id': chunk['id'],
                            'chunk_index': chunk['chunk_index'],
                            'parent_document_id': chunk['parent_document_id'],
                            'parent_document_title': chunk['parent_document_title'],
                            'word_count': chunk['word_count'],
                            'source_file': str(chunk_file)
                        }
                    )
                    all_documents.append(langchain_doc)

            except Exception as e:
                print(f"❌ Error loading {chunk_file}: {e}")

        self.documents = all_documents
        print(f"✅ Loaded {len(all_documents)} total chunks")

    def build_vector_store(self) -> None:
        """Build FAISS vector store from loaded documents"""
        if not self.documents:
            raise ValueError("No documents loaded. Call load_chunked_documents() first.")

        print("🔧 Building FAISS vector store...")

        # Create FAISS vector store
        self.vectorstore = FAISS.from_documents(
            documents=self.documents,
            embedding=self.embeddings
        )

        print(f"✅ Built vector store with {len(self.documents)} document chunks")

    def create_retrieval_qa_chain(self, k: int = 5) -> None:
        """
        Create retrieval QA chain

        Args:
            k: Number of documents to retrieve
        """
        if not self.vectorstore:
            raise ValueError("Vector store not built. Call build_vector_store() first.")

        # Create retriever
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )

        # Create RetrievalQA chain
        self.retrieval_qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt_template}
        )

        print(f"✅ Created retrieval QA chain (k={k})")

    def query(self, question: str) -> Dict:
        """
        Query the RAG system
        Args:
            question: User question
        Returns:
            Dictionary with answer and source information
        """
        if not self.retrieval_qa:
            raise ValueError("QA chain not created. Call create_retrieval_qa_chain() first.")

        print(f"🔍 Processing question: {question}")

        # Get response from QA chain
        result = self.retrieval_qa({"query": question})

        # Extract source information
        source_info = []
        for doc in result.get('source_documents', []):
            source_info.append({
                'chunk_id': doc.metadata.get('chunk_id'),
                'document_title': doc.metadata.get('parent_document_title'),
                'chunk_index': doc.metadata.get('chunk_index'),
                'content_preview': doc.page_content[:100] + "...",
                'word_count': doc.metadata.get('word_count')
            })

        return {
            'question': question,
            'answer': result['result'],
            'source_chunks': source_info,
            'num_sources': len(source_info)
        }

    def save_vector_store(self, save_path: str) -> None:
        """Save the vector store to disk"""
        if not self.vectorstore:
            raise ValueError("No vector store to save")

        self.vectorstore.save_local(save_path)
        print(f"💾 Vector store saved to {save_path}")

    def load_vector_store(self, save_path: str) -> None:
        """Load vector store from disk"""
        self.vectorstore = FAISS.load_local(save_path, self.embeddings)
        print(f"📚 Vector store loaded from {save_path}")


if __name__ == "__main__":
    print("\n STEP 1: Chunking and Saving Documents")
    print("-" * 50)
    chunker = DocumentChunker(
        chunk_size=200,  # 200 words per chunk
        chunk_overlap=30,  # 30 words overlap
        output_dir="./data/who_chunked_docs"
    )
    # chunker.chunk_and_save_documents("./data/knowledge_base")

    print(f"\n STEP 2: LangChain RAG System")
    print("-" * 50)
    rag_system = LangChainRAGSystem(
        model_path='./checkpoints/peft_QA_checkpoints-1752701196/checkpoint-40000',
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    rag_system.load_chunked_documents("./data/who_chunked_docs")
    rag_system.build_vector_store()
    # rag_system.save_vector_store("./data/who_vector_store")
    rag_system.create_retrieval_qa_chain(k=3)   # Retrieve top 3 relevant chunks

    # result = rag_system.query("What are different types of diabetes?")
    result = rag_system.query("How to keep mental health?")
    print(f"Answer: {result['answer']}")
