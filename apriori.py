# Import necessary libraries
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd

# Sample dataset
dataset = [
    ['milk', 'bread', 'eggs'],
    ['milk', 'bread'],
    ['milk', 'eggs','milk', 'bread', 'eggs'],
    ['bread', 'eggs','milk', 'eggs'],
    ['milk', 'bread', 'eggs']
]

# Transaction Encoding
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)

# 1. Identify frequent itemsets
frequent_itemsets = apriori(df, min_support=0.6, use_colnames=True)

# 2. Generate association rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

# Display outputs
print("Frequent Itemsets:\n", frequent_itemsets)
print("\nAssociation Rules:\n", rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])




# MANUAL APRIORI ALGORITHM
from itertools import combinations

# Sample dataset
dataset = [
    ['milk', 'bread', 'eggs'],
    ['milk', 'bread'],
    ['milk', 'eggs'],
    ['bread', 'eggs'],
    ['milk', 'bread', 'eggs']
]

min_support = 0.6
min_confidence = 0.7

# Step 1: Get all unique items
items = sorted(set(item for transaction in dataset for item in transaction))

# Step 2: Generate itemsets and count support
def get_support(itemset, transactions):
    count = sum(1 for t in transactions if set(itemset).issubset(set(t)))
    return count / len(transactions)

# Step 3: Generate frequent itemsets
def apriori_manual(dataset, min_support):
    freq_itemsets = []
    k = 1
    current_itemsets = [[item] for item in items]
    
    while current_itemsets:
        next_itemsets = []
        for itemset in current_itemsets:
            support = get_support(itemset, dataset)
            if support >= min_support:
                freq_itemsets.append((itemset, support))
                for item in items:
                    new_itemset = sorted(set(itemset + [item]))
                    if len(new_itemset) == k+1 and new_itemset not in next_itemsets:
                        next_itemsets.append(new_itemset)
        k += 1
        current_itemsets = next_itemsets
    
    return freq_itemsets

frequent_itemsets = apriori_manual(dataset, min_support)
print("Frequent Itemsets:")
for itemset, support in frequent_itemsets:
    print(f"{itemset}: {support:.2f}")

# Step 4: Generate Association Rules
print("\nAssociation Rules:")
for itemset, support in frequent_itemsets:
    if len(itemset) >= 2:
        for i in range(1, len(itemset)):
            for antecedent in combinations(itemset, i):
                antecedent = list(antecedent)
                consequent = list(set(itemset) - set(antecedent))
                if consequent:
                    conf = get_support(itemset, dataset) / get_support(antecedent, dataset)
                    lift = conf / get_support(consequent, dataset)
                    if conf >= min_confidence:
                        print(f"{antecedent} -> {consequent}, conf: {conf:.2f}, lift: {lift:.2f}")
