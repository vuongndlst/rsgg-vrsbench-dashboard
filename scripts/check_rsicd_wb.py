try:
    with open('/proc/3652660/io') as f:
        print(f.read().strip())
except Exception as e:
    print(e)

