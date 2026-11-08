try:
    import pyarrow.parquet as pq
    print("pyarrow installed successfully!")
    from pathlib import Path
    cache_dir = Path.home() / '.cache' / 'huggingface' / 'hub' / 'datasets--caglarmert--UCM-captions'
    parquet_files = list(cache_dir.rglob('*.parquet'))
    print("Found parquet files:", parquet_files)
    if parquet_files:
        table = pq.read_table(str(parquet_files[0]))
        print("Schema:", table.schema.names)
        print("Num rows:", table.num_rows)
        row0 = {col: table[col][0].as_py() for col in table.schema.names}
        print("Sample row keys:", list(row0.keys()))
        if 'captions' in row0:
            print("Sample captions:", row0['captions'])
        elif 'sentences' in row0:
            print("Sample sentences:", row0['sentences'])
except Exception as e:
    print("Error reading parquet:", e)

