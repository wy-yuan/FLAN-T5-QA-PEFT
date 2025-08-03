# ============================================================================
# WHO GUIDELINES: COLLECT → SAVE LOCALLY → LOAD INTO RAG SYSTEM
# ============================================================================

import requests
import json
import re
import time
import os
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime
import csv

# For PDF processing
try:
    import PyPDF2
    import io

    PDF_AVAILABLE = True
except ImportError:
    print("PyPDF2 not available. Install with: pip install PyPDF2")
    PDF_AVAILABLE = False

# For web scraping
try:
    from bs4 import BeautifulSoup

    BS4_AVAILABLE = True
except ImportError:
    print("BeautifulSoup not available. Install with: pip install beautifulsoup4")
    BS4_AVAILABLE = False

# For RAG system
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig
from peft import PeftModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


class WHOGuidelinesCollector:
    """Collect WHO Guidelines and save them as local documents"""

    def __init__(self, output_dir: str = "who_guidelines_local"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Create subdirectories
        (self.output_dir / "json").mkdir(exist_ok=True)
        (self.output_dir / "text").mkdir(exist_ok=True)
        (self.output_dir / "metadata").mkdir(exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

        print(f"WHO Guidelines will be saved to: {self.output_dir.absolute()}")

    def get_who_guidelines_list(self) -> List[Dict]:
        """Get list of WHO guidelines to collect"""
        # High-priority WHO guidelines with direct URLs
        who_guidelines = [
            {
                "title": "WHO Guidelines on Physical Activity and Sedentary Behaviour",
                "url": "https://www.who.int/publications/i/item/9789240015128",
                "topic": "physical_activity",
                "type": "guidelines",
                "priority": "high"
            },
            {
                "title": "WHO Guidelines for the Treatment of Malaria",
                "url": "https://www.who.int/publications/i/item/guidelines-for-malaria",
                "topic": "malaria",
                "type": "treatment",
                "priority": "high"
            },
            {
                "title": "WHO Mental Health Action Plan",
                "url": "https://www.who.int/publications/i/item/9789241506021",
                "topic": "mental_health",
                "type": "action_plan",
                "priority": "high"
            },
            # WHO Fact Sheets (easier to scrape, reliable content)
            {
                "title": "WHO Fact Sheet: Hypertension",
                "url": "https://www.who.int/news-room/fact-sheets/detail/hypertension",
                "topic": "cardiovascular",
                "type": "fact_sheet",
                "priority": "medium"
            },
            {
                "title": "WHO Fact Sheet: Diabetes",
                "url": "https://www.who.int/news-room/fact-sheets/detail/diabetes",
                "topic": "diabetes",
                "type": "fact_sheet",
                "priority": "medium"
            },
            {
                "title": "WHO Fact Sheet: Cardiovascular Disease",
                "url": "https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)",
                "topic": "cardiovascular",
                "type": "fact_sheet",
                "priority": "medium"
            },
            {
                "title": "WHO Fact Sheet: Mental Disorders",
                "url": "https://www.who.int/news-room/fact-sheets/detail/mental-disorders",
                "topic": "mental_health",
                "type": "fact_sheet",
                "priority": "medium"
            },
            {
                "title": "WHO Fact Sheet: Obesity and Overweight",
                "url": "https://www.who.int/news-room/fact-sheets/detail/obesity-and-overweight",
                "topic": "nutrition",
                "type": "fact_sheet",
                "priority": "medium"
            },
            {
                "title": "WHO Fact Sheet: Cancer",
                "url": "https://www.who.int/news-room/fact-sheets/detail/cancer",
                "topic": "oncology",
                "type": "fact_sheet",
                "priority": "medium"
            },
            {
                "title": "WHO Fact Sheet: Tuberculosis",
                "url": "https://www.who.int/news-room/fact-sheets/detail/tuberculosis",
                "topic": "infectious_disease",
                "type": "fact_sheet",
                "priority": "medium"
            }
        ]

        return who_guidelines

    def extract_content_from_url(self, url: str, title: str) -> Optional[Dict]:
        """Extract content from URL"""
        try:
            print(f"  Fetching: {url}")
            response = self.session.get(url, timeout=15)

            if response.status_code != 200:
                print(f"  ❌ Failed to fetch {url}: HTTP {response.status_code}")
                return None

            # Check if it's a PDF
            content_type = response.headers.get('content-type', '').lower()
            if 'pdf' in content_type:
                return self.extract_pdf_content(response.content, url, title)
            else:
                return self.extract_html_content(response.text, url, title)

        except Exception as e:
            print(f"  ❌ Error extracting from {url}: {e}")
            return None

    def extract_html_content(self, html: str, url: str, title: str) -> Dict:
        """Extract content from HTML page"""
        if not BS4_AVAILABLE:
            # Fallback: basic text extraction
            text = re.sub(r'<[^>]+>', ' ', html)
            text = re.sub(r'\s+', ' ', text).strip()
            return {
                'title': title,
                'content': text[:5000],  # Limit content
                'url': url,
                'extraction_method': 'basic_text'
            }

        soup = BeautifulSoup(html, 'html.parser')

        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            element.decompose()

        # Try to find main content area
        main_content = None
        content_selectors = [
            'main',
            '.main-content',
            '.content',
            'article',
            '.page-content',
            '.fact-sheet-content',
            '.publication-content',
            '#main-content'
        ]

        for selector in content_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                break

        if not main_content:
            main_content = soup.find('body')

        # Extract text
        if main_content:
            # Get text with some structure preserved
            text_content = []

            # Extract headings and paragraphs
            for element in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li']):
                text = element.get_text(strip=True)
                if text and len(text) > 10:
                    if element.name.startswith('h'):
                        text_content.append(f"\n## {text}\n")
                    else:
                        text_content.append(text)

            content = "\n".join(text_content)
        else:
            content = soup.get_text(separator=' ', strip=True)

        # Clean up content
        content = re.sub(r'\n\s*\n', '\n\n', content)  # Remove multiple newlines
        content = re.sub(r' +', ' ', content)  # Remove multiple spaces
        content = content.strip()

        # Extract key facts or summary if available
        key_facts = []
        facts_section = soup.find(['div', 'section'], class_=re.compile(r'key.?facts|summary|highlights'))
        if facts_section:
            for li in facts_section.find_all('li'):
                fact = li.get_text(strip=True)
                if fact:
                    key_facts.append(fact)

        return {
            'title': title,
            'content': content,
            'key_facts': key_facts,
            'url': url,
            'extraction_method': 'beautifulsoup',
            'word_count': len(content.split()),
            'extracted_at': datetime.now().isoformat()
        }

    def extract_pdf_content(self, pdf_bytes: bytes, url: str, title: str) -> Dict:
        """Extract content from PDF"""
        if not PDF_AVAILABLE:
            return {
                'title': title,
                'content': f"PDF content from {url} - PDF extraction not available",
                'url': url,
                'extraction_method': 'pdf_unavailable'
            }

        try:
            pdf_file = io.BytesIO(pdf_bytes)
            pdf_reader = PyPDF2.PdfReader(pdf_file)

            text_content = []
            pages_processed = 0
            max_pages = min(50, len(pdf_reader.pages))  # Limit to 50 pages

            for page_num in range(max_pages):
                try:
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    if text.strip():
                        text_content.append(text)
                        pages_processed += 1
                except Exception as e:
                    print(f"    Error extracting page {page_num}: {e}")
                    continue

            content = "\n".join(text_content)

            # Clean PDF text
            content = re.sub(r'\n\s*\n', '\n\n', content)
            content = re.sub(r' +', ' ', content)
            content = content.strip()

            return {
                'title': title,
                'content': content,
                'url': url,
                'extraction_method': 'pypdf2',
                'pages_processed': pages_processed,
                'total_pages': len(pdf_reader.pages),
                'word_count': len(content.split()),
                'extracted_at': datetime.now().isoformat()
            }

        except Exception as e:
            print(f"    Error processing PDF: {e}")
            return {
                'title': title,
                'content': f"Error extracting PDF content from {url}: {e}",
                'url': url,
                'extraction_method': 'pdf_error'
            }

    def save_guideline_locally(self, guideline_data: Dict, guideline_info: Dict) -> Dict:
        """Save individual guideline to local files"""
        # Create safe filename
        safe_title = re.sub(r'[^\w\s-]', '', guideline_data['title'])
        safe_title = re.sub(r'[-\s]+', '_', safe_title)
        filename_base = safe_title[:50]  # Limit filename length

        # Add topic for organization
        topic = guideline_info.get('topic', 'general')

        # Generate unique ID
        guideline_id = f"who_{topic}_{hash(guideline_data['url']) % 1000000}"

        # Prepare complete guideline data
        complete_data = {
            'id': guideline_id,
            'metadata': {
                'topic': topic,
                'type': guideline_info.get('type', 'unknown'),
                'priority': guideline_info.get('priority', 'medium'),
                'collection_date': datetime.now().isoformat(),
                'source': 'who_official'
            },
            **guideline_data
        }

        # Save as JSON
        json_file = self.output_dir / "json" / f"{filename_base}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(complete_data, f, indent=2, ensure_ascii=False)

        # Save as plain text (for easy reading)
        text_file = self.output_dir / "text" / f"{filename_base}.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(f"Title: {complete_data['title']}\n")
            f.write(f"URL: {complete_data['url']}\n")
            f.write(f"Topic: {topic}\n")
            f.write(f"Extracted: {complete_data.get('extracted_at', 'Unknown')}\n")
            f.write("=" * 50 + "\n\n")
            f.write(complete_data['content'])

            if complete_data.get('key_facts'):
                f.write("\n\nKey Facts:\n")
                for fact in complete_data['key_facts']:
                    f.write(f"• {fact}\n")

        # Save metadata
        metadata_file = self.output_dir / "metadata" / f"{filename_base}_meta.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump({
                'id': guideline_id,
                'title': complete_data['title'],
                'topic': topic,
                'url': complete_data['url'],
                'files': {
                    'json': str(json_file),
                    'text': str(text_file),
                    'metadata': str(metadata_file)
                },
                'extraction_info': {
                    'method': complete_data.get('extraction_method', 'unknown'),
                    'word_count': complete_data.get('word_count', 0),
                    'extracted_at': complete_data.get('extracted_at')
                }
            }, f, indent=2)

        print(f"  ✅ Saved: {json_file.name}")

        return {
            'guideline_id': guideline_id,
            'files': {
                'json': str(json_file),
                'text': str(text_file),
                'metadata': str(metadata_file)
            },
            'data': complete_data
        }

    def collect_and_save_all_guidelines(self) -> Dict:
        """Collect all guidelines and save them locally"""
        print("🔄 Starting WHO Guidelines collection...")

        guidelines_list = self.get_who_guidelines_list()
        print(f"📋 Found {len(guidelines_list)} guidelines to collect")

        results = {
            'successful': [],
            'failed': [],
            'summary': {},
            'collection_info': {
                'start_time': datetime.now().isoformat(),
                'total_guidelines': len(guidelines_list),
                'output_directory': str(self.output_dir.absolute())
            }
        }

        for i, guideline_info in enumerate(guidelines_list):
            print(f"\n📖 Processing {i + 1}/{len(guidelines_list)}: {guideline_info['title']}")

            try:
                # Extract content from URL
                guideline_data = self.extract_content_from_url(
                    guideline_info['url'],
                    guideline_info['title']
                )

                if guideline_data:
                    # Save locally
                    save_result = self.save_guideline_locally(guideline_data, guideline_info)
                    results['successful'].append(save_result)

                    print(f"  📊 Word count: {guideline_data.get('word_count', 'Unknown')}")
                else:
                    results['failed'].append({
                        'guideline_info': guideline_info,
                        'error': 'Failed to extract content'
                    })
                    print(f"  ❌ Failed to extract content")

            except Exception as e:
                results['failed'].append({
                    'guideline_info': guideline_info,
                    'error': str(e)
                })
                print(f"  ❌ Error: {e}")

            time.sleep(2)

        # Generate summary
        results['collection_info']['end_time'] = datetime.now().isoformat()
        results['collection_info']['successful_count'] = len(results['successful'])
        results['collection_info']['failed_count'] = len(results['failed'])

        # Topic summary
        topic_counts = {}
        total_words = 0
        for success in results['successful']:
            topic = success['data']['metadata']['topic']
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
            total_words += success['data'].get('word_count', 0)

        results['summary'] = {
            'topics_collected': topic_counts,
            'total_words': total_words,
            'average_words_per_guideline': total_words / len(results['successful']) if results['successful'] else 0
        }

        # Save collection summary
        summary_file = self.output_dir / "collection_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        # Create CSV index
        self.create_guidelines_index(results['successful'])

        print(f"\n🎉 Collection complete!")
        print(f"✅ Successfully collected: {len(results['successful'])} guidelines")
        print(f"❌ Failed: {len(results['failed'])} guidelines")
        print(f"📊 Total words: {results['summary']['total_words']:,}")
        print(f"📁 Saved to: {self.output_dir.absolute()}")

        return results

    def create_guidelines_index(self, successful_guidelines: List[Dict]):
        """Create a CSV index of all collected guidelines"""
        index_file = self.output_dir / "guidelines_index.csv"

        with open(index_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['id', 'title', 'topic', 'type', 'url', 'word_count', 'json_file', 'text_file']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for guideline in successful_guidelines:
                data = guideline['data']
                writer.writerow({
                    'id': data['id'],
                    'title': data['title'],
                    'topic': data['metadata']['topic'],
                    'type': data['metadata']['type'],
                    'url': data['url'],
                    'word_count': data.get('word_count', 0),
                    'json_file': guideline['files']['json'],
                    'text_file': guideline['files']['text']
                })

        print(f"📇 Created guidelines index: {index_file}")


class RAGSystem:
    """RAG System that loads knowledge from local documents"""

    def __init__(self, model_path: str, guidelines_dir: str = "guidelines_local"):
        """Initialize RAG system with local knowledge base"""
        # Load model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-base')
        peft_model_base = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base", torch_dtype=torch.bfloat16)
        self.model = PeftModel.from_pretrained(peft_model_base, model_path, torch_dtype=torch.bfloat16, is_trainable=False).to(self.device)
        # Initialize embedding model
        # self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        # self.embedding_model = SentenceTransformer('dmis-lab/biobert-base-cased-v1.1')  # need to upgrade torch>=2.6
        self.embedding_model = SentenceTransformer('all-mpnet-base-v2')

        # Local guidelines storage
        self.guidelines_dir = Path(guidelines_dir)
        self.guidelines = []
        self.document_embeddings = None
        self.faiss_index = None

        print(f"RAGSystem initialized")
        print(f"Guidelines directory: {self.guidelines_dir.absolute()}")

    def load_local_guidelines(self) -> bool:
        """Load guidelines from local JSON files"""
        json_dir = self.guidelines_dir / "json"

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
                    self.guidelines.append(guideline_data)
                    loaded_count += 1
                else:
                    print(f"⚠️ Skipping invalid file: {json_file.name}")

            except Exception as e:
                print(f"❌ Error loading {json_file.name}: {e}")

        print(f"✅ Successfully loaded {loaded_count} WHO guidelines")

        if self.guidelines:
            self.build_retrieval_index()
            return True
        else:
            return False

    def build_retrieval_index(self):
        """Build FAISS index for fast retrieval"""
        if not self.guidelines:
            print("❌ No guidelines loaded for indexing")
            return

        print("🔧 Building retrieval index...")

        # Prepare documents for embedding
        documents_for_embedding = []

        for guideline in self.guidelines:
            # Main document content
            main_content = f"Title: {guideline['title']}. Content: {guideline['content']}"
            documents_for_embedding.append(main_content)

            # Add key facts if available
            if guideline.get('key_facts'):
                for fact in guideline['key_facts']:
                    fact_content = f"WHO Key Fact: {fact}"
                    documents_for_embedding.append(fact_content)

                    # Add fact as separate retrievable item
                    fact_item = {
                        'id': f"{guideline['id']}_fact_{len(documents_for_embedding)}",
                        'title': f"{guideline['title']} - Key Fact",
                        'content': fact,
                        'parent_guideline': guideline['id'],
                        'url': guideline['url'],
                        'metadata': guideline['metadata'],
                        'type': 'key_fact'
                    }
                    self.guidelines.append(fact_item)

        # Create embeddings
        print(f"🧮 Creating embeddings for {len(documents_for_embedding)} documents...")
        self.document_embeddings = self.embedding_model.encode(
            documents_for_embedding,
            show_progress_bar=True,
            batch_size=32
        )

        # Build FAISS index
        dimension = self.document_embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)

        # Normalize for cosine similarity
        faiss.normalize_L2(self.document_embeddings)
        self.faiss_index.add(self.document_embeddings.astype('float32'))

        print(f"✅ Built FAISS index with {len(documents_for_embedding)} documents")

    def retrieve_relevant_guidelines(self, query: str, top_k: int = 3) -> List[Dict]:
        """Retrieve relevant knowledge for query"""
        if self.faiss_index is None:
            print("!!! No retrieval index available")
            return []

        # Encode query
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)

        # Search
        scores, indices = self.faiss_index.search(query_embedding.astype('float32'), top_k)

        # Return results
        retrieved = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.guidelines):
                guideline = self.guidelines[idx].copy()
                guideline['retrieval_score'] = float(score)
                retrieved.append(guideline)

        return retrieved

    def generate_answer(self, question: str, retrieved_guidelines: List[Dict], max_length: int = 200) -> str:
        """Generate answer based on knowledge base"""
        if not retrieved_guidelines:
            context = "No relevant knowledge found."
        else:
            context_parts = []
            for guideline in retrieved_guidelines:
                context_part = f"WHO Guideline '{guideline['title']}': {guideline['content']}"
                context_parts.append(context_part)
            context = " ".join(context_parts)

        # Create prompt
        # prompt = f"Based on official World Health Organization guidelines:\n\nContext: {context}\n\n
        # Question: {question}\nAnswer according to WHO recommendations:"
        # based on the information if relevant: {context}
        prompt = f"""
            Instruction: Answer the following question as a doctor based on the information if relevant: {context}
            Question: {question}
            Answer:
            """
        ge_config_v2 = GenerationConfig(
            max_new_tokens=150,
            num_beams=3,
            early_stopping=False,
            repetition_penalty=1.2,  # Prevents repetition!
            no_repeat_ngram_size=3,  # No 3-gram repetition
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        # Generate answer
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        answer = self.tokenizer.decode(self.model.generate(input_ids=inputs.input_ids, generation_config=ge_config_v2)[0],
            skip_special_tokens=True)
        return answer.strip()

    def rag_pipeline(self, question: str, top_k: int = 3) -> Dict:
        """Complete RAG pipeline"""
        print(f"----------->>>>>>>> Processing question: {question}")

        # Retrieve relevant guidelines
        retrieved = self.retrieve_relevant_guidelines(question, top_k)

        # Generate answer
        answer = self.generate_answer(question, retrieved)

        return {
            'question': question,
            'answer': answer,
            # 'retrieved_guidelines': retrieved,
            'sources': [g['url'] for g in retrieved],
            'evidence_count': len(retrieved)
        }

    def get_guidelines_summary(self) -> Dict:
        """Get summary of loaded guidelines"""
        if not self.guidelines:
            return {'status': 'No guidelines loaded'}

        # Count by topic
        topics = {}
        types = {}
        total_words = 0

        for guideline in self.guidelines:
            if guideline.get('type') == 'key_fact':
                continue  # Skip facts in summary

            topic = guideline.get('metadata', {}).get('topic', 'unknown')
            doc_type = guideline.get('metadata', {}).get('type', 'unknown')

            topics[topic] = topics.get(topic, 0) + 1
            types[doc_type] = types.get(doc_type, 0) + 1
            total_words += guideline.get('word_count', 0)

        return {
            'total_guidelines': len([g for g in self.guidelines if g.get('type') != 'key_fact']),
            'topics': topics,
            'types': types,
            'total_words': total_words,
            'has_retrieval_index': self.faiss_index is not None
        }


# ============================================================================
# COMPLETE WORKFLOW: COLLECT → SAVE → LOAD → QUERY
# ============================================================================
if __name__ == "__main__":
    print("\nSTEP 1: Collecting WHO Guidelines")
    print("-" * 40)

    # collector = WHOGuidelinesCollector(output_dir="./data/knowledge_base")
    # collection_results = collector.collect_and_save_all_guidelines()

    print("\n🔧 STEP 2: Loading Guidelines into RAG System")
    print("-" * 40)
    model_path = './checkpoints/peft_QA_checkpoints-1752701196/checkpoint-40000'
    rag_system = RAGSystem(model_path, guidelines_dir="./data/knowledge_base")
    rag_system.load_local_guidelines()

    summary = rag_system.get_guidelines_summary()
    print(f"📊 System Summary:")
    print(f"   Guidelines loaded: {summary['total_guidelines']}")
    print(f"   Topics: {list(summary['topics'].keys())}")
    print(f"   Total words: {summary['total_words']:,}")

    test_question = "Two types of diabetes?"
    result = rag_system.rag_pipeline(test_question, top_k=1)
    print(result)
