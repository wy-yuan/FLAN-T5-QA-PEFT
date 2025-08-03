import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from datasets import Dataset
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import re
from collections import Counter
import json
from typing import List, Dict, Tuple
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass


class MedicalRAGSystem:
    def __init__(self, model_path: str, knowledge_base_path: str = None):
        """
        Initialize the Medical RAG System

        Args:
            model_path: Path to LoRA-tuned FLAN-T5 model
            knowledge_base_path: Path to medical knowledge base (JSON file)
        """
        # Load your fine-tuned model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path).to(self.device)

        # Initialize embedding model for retrieval
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Initialize knowledge base and retriever
        self.knowledge_base = []
        self.document_embeddings = None
        self.faiss_index = None

        if knowledge_base_path:
            self.load_knowledge_base(knowledge_base_path)
            self.build_retriever()

    def load_knowledge_base(self, knowledge_base_path: str):
        """Load medical knowledge base from JSON file"""
        with open(knowledge_base_path, 'r') as f:
            self.knowledge_base = json.load(f)
        print(f"Loaded {len(self.knowledge_base)} documents into knowledge base")

    def create_sample_knowledge_base(self, save_path: str = "medical_kb.json"):
        """Create a sample medical knowledge base for demonstration"""
        sample_kb = [
            {
                "id": "doc_1",
                "title": "Diabetes Management",
                "content": "Type 2 diabetes is managed through lifestyle modifications including diet control, regular exercise, and medication. Metformin is typically the first-line treatment. Blood glucose monitoring is essential for effective management."
            },
            {
                "id": "doc_2",
                "title": "Hypertension Treatment",
                "content": "Hypertension treatment involves lifestyle changes and medications. ACE inhibitors and diuretics are commonly prescribed. Target blood pressure is typically below 130/80 mmHg for most patients."
            },
        ]

        with open(save_path, 'w') as f:
            json.dump(sample_kb, f, indent=2)

        self.knowledge_base = sample_kb
        print(f"Created sample knowledge base with {len(sample_kb)} documents")
        return save_path

    def build_retriever(self):
        """Build FAISS index for fast retrieval"""
        if not self.knowledge_base:
            print("No knowledge base loaded!")
            return

        # Create embeddings for all documents
        doc_texts = [doc['content'] for doc in self.knowledge_base]
        self.document_embeddings = self.embedding_model.encode(doc_texts)

        # Build FAISS index
        dimension = self.document_embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.document_embeddings)
        self.faiss_index.add(self.document_embeddings.astype('float32'))

        print(f"Built FAISS index with {len(self.knowledge_base)} documents")

    def retrieve_documents(self, query: str, top_k: int = 3) -> List[Dict]:
        """Retrieve relevant documents for a given query"""
        if self.faiss_index is None:
            print("Retriever not built! Building now...")
            self.build_retriever()

        # Encode query
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)

        # Search
        scores, indices = self.faiss_index.search(query_embedding.astype('float32'), top_k)

        # Return retrieved documents with scores
        retrieved_docs = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            doc = self.knowledge_base[idx].copy()
            doc['retrieval_score'] = float(score)
            doc['rank'] = i + 1
            retrieved_docs.append(doc)

        return retrieved_docs

    def generate_answer(self, question: str, context_docs: List[Dict] = None, max_length: int = 150) -> str:
        """Generate answer using the fine-tuned model with optional context"""

        if context_docs:
            # Combine retrieved documents as context
            context = " ".join([doc['content'] for doc in context_docs])
            prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
        else:
            prompt = f"Question: {question}\nAnswer:"

        # Tokenize and generate
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
                temperature=0.7,
                do_sample=True
            )

        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer.strip()

    def rag_pipeline(self, question: str, top_k: int = 3, max_length: int = 150) -> Dict:
        """Complete RAG pipeline: retrieve + generate"""
        # Retrieve relevant documents
        retrieved_docs = self.retrieve_documents(question, top_k)

        # Generate answer with context
        answer = self.generate_answer(question, retrieved_docs, max_length)

        return {
            'question': question,
            'answer': answer,
            'retrieved_docs': retrieved_docs,
            'context_used': " ".join([doc['content'] for doc in retrieved_docs])
        }


class RAGEvaluator:
    def __init__(self):
        self.smoothing = SmoothingFunction().method1

    def compute_bleu(self, predictions: List[str], references: List[str]) -> Dict:
        """Compute BLEU scores"""
        bleu_1_scores = []
        bleu_2_scores = []
        bleu_4_scores = []

        for pred, ref in zip(predictions, references):
            pred_tokens = word_tokenize(pred.lower())
            ref_tokens = [word_tokenize(ref.lower())]

            # BLEU-1
            bleu_1 = sentence_bleu(ref_tokens, pred_tokens, weights=(1,), smoothing_function=self.smoothing)
            bleu_1_scores.append(bleu_1)

            # BLEU-2
            bleu_2 = sentence_bleu(ref_tokens, pred_tokens, weights=(0.5, 0.5), smoothing_function=self.smoothing)
            bleu_2_scores.append(bleu_2)

            # BLEU-4
            bleu_4 = sentence_bleu(ref_tokens, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25),
                                   smoothing_function=self.smoothing)
            bleu_4_scores.append(bleu_4)

        return {
            'bleu_1': np.mean(bleu_1_scores),
            'bleu_2': np.mean(bleu_2_scores),
            'bleu_4': np.mean(bleu_4_scores),
            'bleu_1_std': np.std(bleu_1_scores),
            'bleu_2_std': np.std(bleu_2_scores),
            'bleu_4_std': np.std(bleu_4_scores)
        }

    def compute_exact_match(self, predictions: List[str], references: List[str]) -> Dict:
        """Compute Exact Match score"""
        exact_matches = []

        for pred, ref in zip(predictions, references):
            # Normalize text
            pred_normalized = self.normalize_answer(pred)
            ref_normalized = self.normalize_answer(ref)

            exact_match = 1.0 if pred_normalized == ref_normalized else 0.0
            exact_matches.append(exact_match)

        return {
            'exact_match': np.mean(exact_matches),
            'exact_match_std': np.std(exact_matches)
        }

    def normalize_answer(self, text: str) -> str:
        """Normalize answer for exact match comparison"""

        # Remove articles, extra whitespace, and punctuation
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(text))))

    def compute_f1_score(self, predictions: List[str], references: List[str]) -> Dict:
        """Compute token-level F1 score (SQuAD-style)"""
        f1_scores = []

        for pred, ref in zip(predictions, references):
            pred_tokens = self.normalize_answer(pred).split()
            ref_tokens = self.normalize_answer(ref).split()

            if len(ref_tokens) == 0:
                if len(pred_tokens) == 0:
                    f1_scores.append(1.0)
                else:
                    f1_scores.append(0.0)
                continue

            common_tokens = Counter(pred_tokens) & Counter(ref_tokens)
            num_common = sum(common_tokens.values())

            if num_common == 0:
                f1_scores.append(0.0)
                continue

            precision = num_common / len(pred_tokens) if len(pred_tokens) > 0 else 0
            recall = num_common / len(ref_tokens)
            f1 = 2 * precision * recall / (precision + recall)
            f1_scores.append(f1)

        return {
            'f1_score': np.mean(f1_scores),
            'f1_std': np.std(f1_scores)
        }


class HallucinationEvaluator:
    def __init__(self):
        # May need more sophisticated models
        self.fact_checker_model = SentenceTransformer('all-MiniLM-L6-v2')

    def compute_factscore(self, predictions: List[str], contexts: List[str]) -> Dict:
        """
        Simplified FactScore implementation
        Measures how well predictions are supported by retrieved context
        """
        fact_scores = []

        for pred, context in zip(predictions, contexts):
            # Split prediction into sentences/facts
            pred_sentences = self.split_into_facts(pred)

            if not pred_sentences:
                fact_scores.append(0.0)
                continue

            supported_facts = 0

            for fact in pred_sentences:
                # Check if fact is supported by context
                if self.is_fact_supported(fact, context):
                    supported_facts += 1

            fact_score = supported_facts / len(pred_sentences)
            fact_scores.append(fact_score)

        return {
            'factscore': np.mean(fact_scores),
            'factscore_std': np.std(fact_scores),
            'avg_facts_per_answer': np.mean([len(self.split_into_facts(pred)) for pred in predictions])
        }

    def split_into_facts(self, text: str) -> List[str]:
        """Split text into individual factual claims"""
        # Simple sentence splitting - in practice, use more sophisticated methods
        sentences = re.split(r'[.!?]+', text)
        facts = [s.strip() for s in sentences if len(s.strip()) > 10]
        return facts

    def is_fact_supported(self, fact: str, context: str, threshold: float = 0.7) -> bool:
        """Check if a fact is supported by the context using semantic similarity"""
        if not fact.strip() or not context.strip():
            return False

        # Encode fact and context
        fact_embedding = self.fact_checker_model.encode([fact])

        # Split context into sentences and find best match
        context_sentences = re.split(r'[.!?]+', context)
        context_sentences = [s.strip() for s in context_sentences if len(s.strip()) > 5]

        if not context_sentences:
            return False

        context_embeddings = self.fact_checker_model.encode(context_sentences)

        # Compute similarities
        similarities = cosine_similarity(fact_embedding, context_embeddings)[0]
        max_similarity = np.max(similarities)

        return max_similarity > threshold

    def compute_knowledge_grounded_score(self, predictions: List[str], questions: List[str],
                                         retrieved_docs: List[List[Dict]]) -> Dict:
        """
        Evaluate how well predictions are grounded in retrieved knowledge
        """
        grounding_scores = []
        relevance_scores = []

        for pred, question, docs in zip(predictions, questions, retrieved_docs):
            # Combine all retrieved document content
            full_context = " ".join([doc['content'] for doc in docs])

            # Compute grounding score (how well prediction is supported)
            grounding_score = self.compute_grounding_score(pred, full_context)
            grounding_scores.append(grounding_score)

            # Compute relevance score (how relevant retrieved docs are to question)
            relevance_score = self.compute_relevance_score(question, docs)
            relevance_scores.append(relevance_score)

        return {
            'knowledge_grounding': np.mean(grounding_scores),
            'retrieval_relevance': np.mean(relevance_scores),
            'grounding_std': np.std(grounding_scores),
            'relevance_std': np.std(relevance_scores)
        }

    def compute_grounding_score(self, prediction: str, context: str) -> float:
        """Compute how well prediction is grounded in context"""
        if not prediction.strip() or not context.strip():
            return 0.0

        pred_embedding = self.fact_checker_model.encode([prediction])
        context_embedding = self.fact_checker_model.encode([context])

        similarity = cosine_similarity(pred_embedding, context_embedding)[0][0]
        return float(similarity)

    def compute_relevance_score(self, question: str, retrieved_docs: List[Dict]) -> float:
        """Compute average relevance of retrieved documents to question"""
        if not retrieved_docs:
            return 0.0

        question_embedding = self.fact_checker_model.encode([question])

        relevance_scores = []
        for doc in retrieved_docs:
            doc_embedding = self.fact_checker_model.encode([doc['content']])
            similarity = cosine_similarity(question_embedding, doc_embedding)[0][0]
            relevance_scores.append(similarity)

        return np.mean(relevance_scores)


# Example usage and evaluation
def run_comprehensive_evaluation():
    """Complete evaluation pipeline example"""

    # Initialize RAG system (replace with your actual model path)
    model_path = "your_lora_flan_t5_model_path"  # Replace with actual path
    rag_system = MedicalRAGSystem(model_path)

    # Create sample knowledge base for demonstration
    kb_path = rag_system.create_sample_knowledge_base()
    rag_system.load_knowledge_base(kb_path)
    rag_system.build_retriever()

    # Sample test data
    test_questions = [
        "What is the first-line treatment for type 2 diabetes?",
        "What is the target blood pressure for most patients with hypertension?",
        "When should someone seek immediate medical attention for chest pain?",
        "How can antibiotic resistance be prevented?",
        "Who should get annual flu vaccines?"
    ]

    # Reference answers
    reference_answers = [
        "Metformin is typically the first-line treatment for type 2 diabetes.",
        "Target blood pressure is typically below 130/80 mmHg for most patients.",
        "Immediate medical attention is required for acute chest pain with accompanying symptoms like shortness of breath or sweating.",
        "Antibiotic resistance can be prevented by completing prescribed courses and avoiding unnecessary antibiotic use.",
        "Annual flu vaccines are recommended for most people over 6 months."
    ]

    # Generate predictions using RAG
    predictions = []
    contexts = []
    retrieved_docs_list = []

    print("Generating answers using RAG pipeline...")
    for question in test_questions:
        result = rag_system.rag_pipeline(question)
        predictions.append(result['answer'])
        contexts.append(result['context_used'])
        retrieved_docs_list.append(result['retrieved_docs'])
        print(f"Q: {question}")
        print(f"A: {result['answer']}")
        print("-" * 50)

    # Initialize evaluators
    rag_evaluator = RAGEvaluator()
    hallucination_evaluator = HallucinationEvaluator()

    # Compute all metrics
    print("\nComputing evaluation metrics...")

    # BLEU scores
    bleu_results = rag_evaluator.compute_bleu(predictions, reference_answers)

    # Exact Match
    em_results = rag_evaluator.compute_exact_match(predictions, reference_answers)

    # F1 Score
    f1_results = rag_evaluator.compute_f1_score(predictions, reference_answers)

    # FactScore (Hallucination)
    fact_results = hallucination_evaluator.compute_factscore(predictions, contexts)

    # Knowledge Grounded Evaluation
    kg_results = hallucination_evaluator.compute_knowledge_grounded_score(
        predictions, test_questions, retrieved_docs_list
    )

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print(f"\nBLEU Scores:")
    print(f"  BLEU-1: {bleu_results['bleu_1']:.3f} ± {bleu_results['bleu_1_std']:.3f}")
    print(f"  BLEU-2: {bleu_results['bleu_2']:.3f} ± {bleu_results['bleu_2_std']:.3f}")
    print(f"  BLEU-4: {bleu_results['bleu_4']:.3f} ± {bleu_results['bleu_4_std']:.3f}")

    print(f"\nExact Match:")
    print(f"  EM: {em_results['exact_match']:.3f} ± {em_results['exact_match_std']:.3f}")

    print(f"\nF1 Score:")
    print(f"  F1: {f1_results['f1_score']:.3f} ± {f1_results['f1_std']:.3f}")

    print(f"\nHallucination Metrics:")
    print(f"  FactScore: {fact_results['factscore']:.3f} ± {fact_results['factscore_std']:.3f}")
    print(f"  Avg Facts/Answer: {fact_results['avg_facts_per_answer']:.1f}")

    print(f"\nKnowledge Grounding:")
    print(f"  Grounding Score: {kg_results['knowledge_grounding']:.3f} ± {kg_results['grounding_std']:.3f}")
    print(f"  Retrieval Relevance: {kg_results['retrieval_relevance']:.3f} ± {kg_results['relevance_std']:.3f}")

    # Create summary dataframe
    results_df = pd.DataFrame([
        {'Metric': 'BLEU-1', 'Score': bleu_results['bleu_1'], 'Std': bleu_results['bleu_1_std']},
        {'Metric': 'BLEU-2', 'Score': bleu_results['bleu_2'], 'Std': bleu_results['bleu_2_std']},
        {'Metric': 'BLEU-4', 'Score': bleu_results['bleu_4'], 'Std': bleu_results['bleu_4_std']},
        {'Metric': 'Exact Match', 'Score': em_results['exact_match'], 'Std': em_results['exact_match_std']},
        {'Metric': 'F1 Score', 'Score': f1_results['f1_score'], 'Std': f1_results['f1_std']},
        {'Metric': 'FactScore', 'Score': fact_results['factscore'], 'Std': fact_results['factscore_std']},
        {'Metric': 'Knowledge Grounding', 'Score': kg_results['knowledge_grounding'],
         'Std': kg_results['grounding_std']},
        {'Metric': 'Retrieval Relevance', 'Score': kg_results['retrieval_relevance'],
         'Std': kg_results['relevance_std']}
    ])

    print(f"\n{results_df.to_string(index=False, float_format='%.3f')}")

    return results_df


if __name__ == "__main__":
    # Run the complete evaluation
    results = run_comprehensive_evaluation()