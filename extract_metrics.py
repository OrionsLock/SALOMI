"""Extract key metrics from validation results"""
with open('validation_results.txt', 'r', encoding='utf-8', errors='ignore') as f:
    content = f.read()
    
# Print the full content
print("=== FULL VALIDATION RESULTS ===")
print(content)
print("\n=== KEY METRICS ===")

# Extract key lines
for line in content.split('\n'):
    if any(keyword in line for keyword in ['Correlation', 'BPP', 'Throughput', 'Params', 'Results']):
        print(line)
