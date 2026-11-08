text = open('papers/geochat_text.txt', encoding='utf-8').read()

pos_t7 = text.find("Table 7.")
if pos_t7 != -1:
    print("=== TABLE 7 & AROUND ===")
    print(text[pos_t7-500:pos_t7+1500])

pos_t8 = text.find("Table 8.")
if pos_t8 != -1:
    print("\n=== TABLE 8 & 9 & 10 ===")
    print(text[pos_t8-200:pos_t8+2500])

