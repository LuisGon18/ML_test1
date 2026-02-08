from file_loader.data_loader import DataLoader
from preprocessing import PreprocessingData
from training import ModelTrainer
from model_evaluation import ModelEvaluator

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

        # 4. Modelado
        num_classes = y_train.shape[1]  # Number of columns in one-hot encoded y
        trainer = ModelTrainer(input_size=input_size, num_classes=num_classes)
        trainer.build_model()
        trainer.train(x_train, y_train, epochs=20)

        # 5. Evaluación del modelo
        evaluator = ModelEvaluator(trainer.model)
        results = evaluator.evaluate(x_test, y_test)

        print("Loss:", results["loss"])
        print("Accuracy:", results["accuracy"])
        print("Precision:", results["precision"])
        print("Recall:", results["recall"])
        print("F1:", results["f1"])
        print("Confusion Matrix:\n", results["confusion_matrix"])
        print("Classification Report:\n", results["classification_report"])
