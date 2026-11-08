import pycocoevalcap, os
print("pycocoevalcap path:", pycocoevalcap.__path__)
for p in pycocoevalcap.__path__:
    print("Listing:", p)
    for root, dirs, files in os.walk(p):
        print(" ", root, dirs, files[:5])
