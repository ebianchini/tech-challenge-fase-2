from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.ml_project.config import INTERIM_DIR, RAW_DATA_PATH


def load_raw_dataset(path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Carrega o dataset bruto do caminho informado."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de dados não encontrado: {path}")

    dataset = pd.read_csv(path)
    return dataset


def drop_duplicate_rows(dataset: pd.DataFrame) -> pd.DataFrame:
    """Remove linhas duplicadas e restabelece o índice."""
    cleaned = dataset.drop_duplicates().reset_index(drop=True)
    return cleaned


def save_dataset(dataset: pd.DataFrame, output_path: str | Path) -> Path:
    """Salva um dataset em CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)
    return output_path


def prepare_interim_dataset() -> pd.DataFrame:
    """Carrega, limpa e salva o dataset intermediário para o pipeline DVC."""
    raw_data = load_raw_dataset()
    cleaned_data = drop_duplicate_rows(raw_data)

    interim_path = INTERIM_DIR / "online_shoppers_clean.csv"
    save_dataset(cleaned_data, interim_path)
    return cleaned_data
