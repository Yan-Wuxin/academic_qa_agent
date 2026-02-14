import json

def load_testset(path="testset.json"): ###
    with open(path, "r", encoding="utf-8") as f:
        testset = json.load(f)
    return testset

def evaluate_recall(retriever, testset, k=5):
    hit = 0
    for item in testset:
        query = item["query"]
        gold_texts = item["gold_texts"]
        docs = retriever.invoke(query)
        retriever_texts = [d.page_content for d in docs[:k]]
        success = False
        for gold in gold_texts:
            for ret in retriever_texts:
                if gold[:100] in ret:
                    success = True
        if success:
            hit += 1

    recall = hit / len(testset)
    print(f"Recall@{k}: {recall:.3f}")