
from contextlib import redirect_stdout
from io import StringIO

from backend.services.semantic_cache import SemanticCache
from backend.services.embedding_service import generate_embedding


# --------------------------------------------------
# 1. Evaluation dataset
# --------------------------------------------------

# expected_hit=True means both questions have the same intent.
# expected_hit=False means they require different answers.

test_cases = [

    # Equivalent questions: expected cache HIT
    {
        "category": "Equivalent",
        "cached": "How many sick leaves do employees receive?",
        "query": "What is the annual sick leave allowance?",
        "expected_hit": True
    },
    {
        "category": "Equivalent",
        "cached": "How many sick leaves do employees receive?",
        "query": "How many sick leave days are employees entitled to?",
        "expected_hit": True
    },
    {
        "category": "Equivalent",
        "cached": "How many casual leaves do employees receive?",
        "query": "How many casual leave days can employees take?",
        "expected_hit": True
    },
    {
        "category": "Equivalent",
        "cached": "What are the company's working hours?",
        "query": "What are the standard working hours at the company?",
        "expected_hit": True
    },
    {
        "category": "Equivalent",
        "cached": "How many work from home days are allowed each month?",
        "query": "How many days per month can employees work from home?",
        "expected_hit": True
    },
    {
        "category": "Equivalent",
        "cached": "When are salaries credited?",
        "query": "On which day are employee salaries paid?",
        "expected_hit": True
    },
    {
        "category": "Equivalent",
        "cached": "How long is the lunch break?",
        "query": "What is the duration of the lunch break?",
        "expected_hit": True
    },
    {
        "category": "Equivalent",
        "cached": "Does working from home require manager approval?",
        "query": "Is manager approval needed for work from home?",
        "expected_hit": True
    },
    {
        "category": "Equivalent",
        "cached": "What is the company's standard working schedule?",
        "query": "What is the standard work schedule at the company?",
        "expected_hit": True
    },
    {
        "category": "Equivalent",
        "cached": "When does the company credit monthly salaries?",
        "query": "When are monthly salaries paid by the company?",
        "expected_hit": True
    },

    # Different questions: expected cache MISS
    {
        "category": "Different",
        "cached": "How many sick leaves do employees receive?",
        "query": "How many casual leaves do employees receive?",
        "expected_hit": False
    },
    {
        "category": "Different",
        "cached": "How many casual leaves do employees receive?",
        "query": "How many sick leaves do employees receive?",
        "expected_hit": False
    },
    {
        "category": "Different",
        "cached": "What are the company's working hours?",
        "query": "Which days of the week does the company operate?",
        "expected_hit": False
    },
    {
        "category": "Different",
        "cached": "When are salaries credited?",
        "query": "How much salary do employees receive?",
        "expected_hit": False
    },
    {
        "category": "Different",
        "cached": "How many work from home days are allowed each month?",
        "query": "Does working from home require manager approval?",
        "expected_hit": False
    },
    {
        "category": "Different",
        "cached": "How long is the lunch break?",
        "query": "What are the company's working hours?",
        "expected_hit": False
    },
    {
        "category": "Different",
        "cached": "How many sick leaves do employees receive?",
        "query": "What are the company's working hours?",
        "expected_hit": False
    },
    {
        "category": "Different",
        "cached": "When are salaries credited?",
        "query": "How many work from home days are allowed?",
        "expected_hit": False
    },
    {
        "category": "Different",
        "cached": "What are the company's working hours?",
        "query": "How many casual leaves do employees receive?",
        "expected_hit": False
    },
    {
        "category": "Different",
        "cached": "How many work from home days are allowed each month?",
        "query": "When are salaries credited?",
        "expected_hit": False
    },

    
    # Hard negatives: similar wording, different meaning

    {
        "category": "Hard Negative",
        "cached": "How many sick leaves do employees receive?",
        "query": "Do employees receive 10 sick leaves per year?",
        "expected_hit": False
    },
    {
        "category": "Hard Negative",
        "cached": "How many sick leaves do employees receive?",
        "query": "Do employees receive no sick leaves?",
        "expected_hit": False
    },
    {
        "category": "Hard Negative",
        "cached": "How many sick leaves do employees receive?",
        "query": "How many casual leaves do employees receive?",
        "expected_hit": False
    },
    {
        "category": "Hard Negative",
        "cached": "What are the company's working hours?",
        "query": "Are the company's working hours 10 AM to 7 PM?",
        "expected_hit": False
    },
    {
        "category": "Hard Negative",
        "cached": "How long is the lunch break?",
        "query": "Is the lunch break 30 minutes?",
        "expected_hit": False
    },
    {
        "category": "Hard Negative",
        "cached": "When are salaries credited?",
        "query": "Are salaries credited on the first day of the month?",
        "expected_hit": False
    },
    {
        "category": "Hard Negative",
        "cached": "How many work from home days are allowed each month?",
        "query": "Can employees work from home every day?",
        "expected_hit": False
    },
    {
        "category": "Hard Negative",
        "cached": "Does working from home require manager approval?",
        "query": "Is manager approval not required for working from home?",
        "expected_hit": False
    },
    {
        "category": "Hard Negative",
        "cached": "How many casual leaves do employees receive?",
        "query": "Are employees entitled to 20 casual leaves?",
        "expected_hit": False
    },
    {
        "category": "Hard Negative",
        "cached": "How many sick leaves do employees receive?",
        "query": "Are employees prohibited from taking sick leave?",
        "expected_hit": False
    }

]


# --------------------------------------------------
# 2. Generate embeddings once
# --------------------------------------------------

print("\nGenerating embeddings for evaluation dataset...")

all_questions = set()

for case in test_cases:
    all_questions.add(case["cached"])
    all_questions.add(case["query"])

embedding_map = {}

for question in all_questions:
    embedding_map[question] = generate_embedding(question)

print("Unique questions embedded:", len(embedding_map))


# --------------------------------------------------
# 3. Evaluate different thresholds
# --------------------------------------------------

thresholds = [0.65, 0.70, 0.75, 0.80, 0.85]

results_by_threshold = {}

for threshold in thresholds:

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    case_results = []

    for case in test_cases:

        cache = SemanticCache(
            similarity_threshold=threshold,
            ttl=3600
        )

        # Add the original question to cache
        cache.add(
            question=case["cached"],
            embedding=embedding_map[case["cached"]],
            answer="Sample cached answer",
            sources=[{"page": 1}]
        )

        # Search using the new question
        # Suppress debug prints during evaluation
        with redirect_stdout(StringIO()):

            result = cache.search(
                embedding_map[case["query"]],
                case["query"]
            )

        predicted_hit = result is not None
        expected_hit = case["expected_hit"]

        # Confusion matrix calculations
        if expected_hit and predicted_hit:
            tp += 1

        elif not expected_hit and predicted_hit:
            fp += 1

        elif not expected_hit and not predicted_hit:
            tn += 1

        elif expected_hit and not predicted_hit:
            fn += 1

        case_results.append({
            "query": case["query"],
            "expected": expected_hit,
            "predicted": predicted_hit,
            "correct": expected_hit == predicted_hit
        })

    # Calculate evaluation metrics
    total = tp + fp + tn + fn

    precision = (
        tp / (tp + fp) * 100
        if tp + fp > 0 else 0
    )

    recall = (
        tp / (tp + fn) * 100
        if tp + fn > 0 else 0
    )

    accuracy = (
        (tp + tn) / total * 100
        if total > 0 else 0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0
    )

    results_by_threshold[threshold] = {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "f1": f1,
        "cases": case_results
    }


# --------------------------------------------------
# 4. Display comparison
# --------------------------------------------------

print("\n========== THRESHOLD COMPARISON ==========")

print(
    f"{'Threshold':<12}"
    f"{'Precision':<13}"
    f"{'Recall':<12}"
    f"{'Accuracy':<13}"
    f"{'F1 Score':<12}"
)

for threshold, metrics in results_by_threshold.items():

    print(
        f"{threshold:<12.2f}"
        f"{metrics['precision']:<13.2f}"
        f"{metrics['recall']:<12.2f}"
        f"{metrics['accuracy']:<13.2f}"
        f"{metrics['f1']:<12.2f}"
    )


# --------------------------------------------------
# 5. Detailed results at current threshold
# --------------------------------------------------

selected_threshold = 0.85

selected_results = results_by_threshold[selected_threshold]

print(f"\n========== DETAILED RESULTS: {selected_threshold} ==========")

for index, result in enumerate(selected_results["cases"], start=1):

    expected = "HIT" if result["expected"] else "MISS"
    predicted = "HIT" if result["predicted"] else "MISS"
    status = "CORRECT" if result["correct"] else "INCORRECT"

    print(f"\nTest {index}")
    print("Question:", result["query"])
    print("Expected:", expected)
    print("Predicted:", predicted)
    print("Evaluation:", status)


# --------------------------------------------------
# 6. Confusion matrix
# --------------------------------------------------

print("\n========== CONFUSION MATRIX ==========")

print("True Positives :", selected_results["TP"])
print("False Positives:", selected_results["FP"])
print("True Negatives :", selected_results["TN"])
print("False Negatives:", selected_results["FN"])

print("\nEvaluation completed successfully.")
