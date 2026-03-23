import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone_store import search_pinecone
from main import build_messages
import ollama
import json

load_dotenv()

MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL  = "llama3.2"

# ─────────────────────────────────────────────
# Test Dataset
# 20 question-answer pairs to evaluate quality
# ─────────────────────────────────────────────
TEST_DATASET = [
    {
        "question": "When do licenses need to be renewed?",
        "ground_truth": "Software licenses must be renewed 30 days before expiration."
    },
    {
        "question": "Who approves renewals exceeding $10,000?",
        "ground_truth": "Renewals exceeding $10,000 require VP approval before processing."
    },
    {
        "question": "What is the grace period for licenses under active renewal?",
        "ground_truth": "Grace periods of 7 days are granted for licenses under active renewal."
    },
    {
        "question": "What happens to expired licenses?",
        "ground_truth": "Expired licenses must be removed from all systems within 48 hours."
    },
    {
        "question": "Who is responsible for initiating renewal requests?",
        "ground_truth": "The procurement team is responsible for initiating renewal requests."
    },
    {
        "question": "What are employees prohibited from installing?",
        "ground_truth": "Employees are prohibited from installing unlicensed software on company devices."
    },
    {
        "question": "Where must license keys be stored?",
        "ground_truth": "License keys must be stored in the central license management system."
    },
    {
        "question": "When are annual audits conducted?",
        "ground_truth": "Annual audits are conducted every December to verify license compliance."
    },
    {
        "question": "How quickly must violations be reported?",
        "ground_truth": "Violations must be reported to the IT compliance team within 24 hours."
    },
    {
        "question": "How often must license usage reports be submitted?",
        "ground_truth": "License usage reports must be submitted to management every quarter."
    },
]

def get_answer(question, model):
    """Get answer from LicenseBot for a question"""
    results = search_pinecone(question, model)
    context = ""
    if results:
        for r in results:
            context += f"\n---\n{r['content']}\n"
    else:
        context = "No relevant context found."

    messages = build_messages([], context, question)
    response = ollama.chat(model=LLM_MODEL, messages=messages)
    answer   = response["message"]["content"]
    contexts = [r["content"] for r in results]
    return answer, contexts

def evaluate():
    print("="*60)
    print("  LicenseBot — RAG Evaluation")
    print("="*60)

    model   = SentenceTransformer(MODEL_NAME)
    results = []
    scores  = {
        "faithfulness":    [],
        "answer_relevancy": [],
        "total_questions": len(TEST_DATASET)
    }

    for i, item in enumerate(TEST_DATASET):
        question    = item["question"]
        ground_truth = item["ground_truth"]

        print(f"\n[{i+1}/{len(TEST_DATASET)}] {question}")

        answer, contexts = get_answer(question, model)

        # Simple faithfulness check
        # Does the answer contain key phrases from ground truth?
        gt_words     = set(ground_truth.lower().split())
        answer_words = set(answer.lower().split())
        overlap      = len(gt_words & answer_words) / len(gt_words)
        faithfulness = round(overlap, 2)

        # Simple relevancy check
        # Does the answer address the question topic?
        q_words      = set(question.lower().split())
        relevancy    = round(len(q_words & answer_words) / len(q_words), 2)

        scores["faithfulness"].append(faithfulness)
        scores["answer_relevancy"].append(relevancy)

        result = {
            "question":     question,
            "answer":       answer[:100] + "...",
            "ground_truth": ground_truth,
            "faithfulness": faithfulness,
            "relevancy":    relevancy,
            "pass":         faithfulness > 0.3 and relevancy > 0.2
        }
        results.append(result)

        status = "✅" if result["pass"] else "❌"
        print(f"   {status} Faithfulness: {faithfulness} | Relevancy: {relevancy}")

    # Summary
    avg_faithfulness = round(
        sum(scores["faithfulness"]) / len(scores["faithfulness"]), 2
    )
    avg_relevancy = round(
        sum(scores["answer_relevancy"]) / len(scores["answer_relevancy"]), 2
    )
    passed = sum(1 for r in results if r["pass"])

    print("\n" + "="*60)
    print("  EVALUATION SUMMARY")
    print("="*60)
    print(f"  Total questions:     {len(TEST_DATASET)}")
    print(f"  Passed:              {passed}/{len(TEST_DATASET)}")
    print(f"  Avg Faithfulness:    {avg_faithfulness}")
    print(f"  Avg Relevancy:       {avg_relevancy}")
    print("="*60)

    # Save results to JSON
    output = {
        "summary": {
            "total":            len(TEST_DATASET),
            "passed":           passed,
            "avg_faithfulness": avg_faithfulness,
            "avg_relevancy":    avg_relevancy
        },
        "results": results
    }

    with open("evaluation_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\n📊 Full results saved to evaluation_results.json")

if __name__ == "__main__":
    evaluate()