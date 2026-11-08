with open('/proc/3625041/io') as f:
    for line in f:
        if 'write_bytes' in line:
            print(line.strip())

