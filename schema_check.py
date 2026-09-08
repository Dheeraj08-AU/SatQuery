import datasets

def check_schema(dataset_name, config_name=None):
    print(f"\n============================================================")
    print(f"Checking schema for: {dataset_name}")
    try:
        if config_name:
            ds = datasets.load_dataset(dataset_name, config_name, streaming=True)
        else:
            ds = datasets.load_dataset(dataset_name, streaming=True)
            
        # Get the first split (usually 'train')
        split_name = list(ds.keys())[0]
        split_ds = ds[split_name]
        
        # Get one example
        example = next(iter(split_ds))
        
        print("Feature Schema:")
        for key, value in split_ds.features.items():
            print(f" - {key}: {value}")
            
        print("\nExample data (truncated):")
        for key, value in example.items():
            val_str = str(value)
            if len(val_str) > 100:
                val_str = val_str[:100] + "... [TRUNCATED]"
            print(f" - {key}: {val_str}")
            
    except Exception as e:
        print(f"Error loading {dataset_name}: {e}")
    print(f"============================================================\n")

if __name__ == "__main__":
    check_schema("Hermanni/sen12mscr")
    check_schema("BIFOLD-BigEarthNetv2-0/BigEarthNet.txt")
