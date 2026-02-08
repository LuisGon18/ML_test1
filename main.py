from file_loader.data_loader import DataLoader

if __name__ == "__main__":
    # 1. Create the instance (this runs __init__ and sets the path)
    loader = DataLoader("iris.csv")

    # 2. Call the method from that instance
    df = loader.load_csv()

    # 3. Validate and print
    if loader.validate_data():
        print(df.head())
