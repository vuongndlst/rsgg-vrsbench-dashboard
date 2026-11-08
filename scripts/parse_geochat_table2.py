text = open('papers/geochat_text.txt', encoding='utf-8').read()

pos_t2 = text.find("Table 2.")
if pos_t2 != -1:
    print(text[pos_t2-200:pos_t2+1000])

# Also check citations for datasets
print("\n=== CITATIONS FOR DATASETS ===")
for ref in ['DOTA', 'DIOR', 'FAIR1M', 'SAMRS', 'NWPU', 'LRBEN', 'HRBEN', 'AID', 'UCMerced', 'FloodNet']:
    print(f"\n--- {ref} ---")
    matches = [line for line in text.splitlines() if ref in line]
    for l in matches[:3]:
        print(" ", l)

