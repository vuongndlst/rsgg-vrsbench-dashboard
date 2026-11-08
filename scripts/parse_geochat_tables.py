import re

text = open('papers/geochat_text.txt', encoding='utf-8').read()

print("=== TABLE 1 & 2 DETAILS ===")
pos_t1 = text.find("Table 1.")
if pos_t1 != -1:
    print(text[pos_t1:pos_t1+1500])

pos_t2 = text.find("Table 2.")
if pos_t2 != -1:
    print(text[pos_t2:pos_t2+1500])

print("\n=== EXPERIMENTS SECTION & EVAL DATASETS ===")
pos_exp = text.find("5. Experiments")
if pos_exp != -1:
    print(text[pos_exp:pos_exp+3000])

