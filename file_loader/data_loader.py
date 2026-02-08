import pandas as pd
from pathlib import Path
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(self, file_name: str):
        """
        Initializes the loader.
        It automatically finds the root project directory from this script.
        """
        self.base_dir = Path(__file__).resolve().parent.parent
        self.file_path = self.base_dir / file_name
        self.data = None

    def load_csv(self) -> pd.DataFrame:
        """Loads the CSV file into a pandas DataFrame."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"No se encontró el archivo en: {self.file_path}")

        try:
            self.data = pd.read_csv(self.file_path)
            logger.info(f"Archivo {self.file_path.name} cargado exitosamente.")
            return self.data
        except Exception as e:
            logger.error(f"Error al cargar el CSV: {e}")
            raise

    def validate_data(self, required_columns: list = None) -> bool:
        """
        Performs basic validation:
        1. Checks if data is loaded.
        2. Checks if the DataFrame is empty.
        3. (Optional) Checks for specific required columns.
        """
        if self.data is None:
            logger.warning("No hay datos cargados para validar.")
            return False

        if self.data.empty:
            logger.error("El archivo CSV está vacío.")
            return False

        if required_columns:
            missing = [col for col in required_columns if col not in self.data.columns]
            if missing:
                logger.error(f"Faltan columnas requeridas: {missing}")
                return False

        logger.info("Validación completada con éxito.")
        return True