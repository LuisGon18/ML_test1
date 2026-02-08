from file_loader.data_loader import DataLoader
from preprocessing import PreprocessingData

if __name__ == "__main__":
    # 1) Cargar el CSV
    loader = DataLoader("iris.csv")
    df = loader.load_csv()

    if loader.validate_data():
        print(df.head())

        # 2) Separar X (features) y y (target)
        # Ajustá el nombre de la columna target según tu CSV:
        # comúnmente: "species" o "variety"
        y = df["species"]
        X = df.drop(columns=["species"])

        # 3) Preprocesar
        pp = PreprocessingData(test_size=0.10, random_state=101)
        x_train, x_test, y_train, y_test, input_size = pp.fit_transform(X, y)

        # 4) Validar outputs
        print("x_train:", x_train.shape)
        print("x_test :", x_test.shape)
        print("y_train:", y_train.shape)
        print("y_test :", y_test.shape)
        print("input_size:", input_size)
