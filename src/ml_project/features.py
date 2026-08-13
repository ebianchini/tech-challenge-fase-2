from __future__ import annotations

import numpy as np
import pandas as pd


def add_session_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """Cria variáveis derivadas a partir da duração e número de páginas por sessão."""
    feature_frame = dataset.copy()

    feature_frame["TotalSessionTime"] = (
        feature_frame["Administrative_Duration"]
        + feature_frame["Informational_Duration"]
        + feature_frame["ProductRelated_Duration"]
    )

    feature_frame["TotalPagesVisited"] = (
        feature_frame["Administrative"]
        + feature_frame["Informational"]
        + feature_frame["ProductRelated"]
    )

    total_time = feature_frame["TotalSessionTime"].to_numpy(dtype=float)
    administrative_duration = feature_frame["Administrative_Duration"].to_numpy(dtype=float)
    informational_duration = feature_frame["Informational_Duration"].to_numpy(dtype=float)
    product_duration = feature_frame["ProductRelated_Duration"].to_numpy(dtype=float)

    feature_frame["AdministrativeTimeRatio"] = np.divide(
        administrative_duration,
        total_time,
        out=np.zeros_like(administrative_duration, dtype=float),
        where=total_time > 0,
    )
    feature_frame["InformationalTimeRatio"] = np.divide(
        informational_duration,
        total_time,
        out=np.zeros_like(informational_duration, dtype=float),
        where=total_time > 0,
    )
    feature_frame["ProductRelatedTimeRatio"] = np.divide(
        product_duration,
        total_time,
        out=np.zeros_like(product_duration, dtype=float),
        where=total_time > 0,
    )

    product_time_ratio = feature_frame["ProductRelatedTimeRatio"].to_numpy(dtype=float)
    administrative_counts = feature_frame["Administrative"].to_numpy(dtype=float)
    informational_counts = feature_frame["Informational"].to_numpy(dtype=float)
    product_counts = feature_frame["ProductRelated"].to_numpy(dtype=float)

    feature_frame["AdministrativeRatio"] = np.divide(
        administrative_counts,
        product_time_ratio,
        out=np.zeros_like(administrative_counts, dtype=float),
        where=product_time_ratio != 0,
    )
    feature_frame["InformationalRatio"] = np.divide(
        informational_counts,
        product_time_ratio,
        out=np.zeros_like(informational_counts, dtype=float),
        where=product_time_ratio != 0,
    )
    feature_frame["ProductRelatedRatio"] = np.divide(
        product_counts,
        product_time_ratio,
        out=np.zeros_like(product_counts, dtype=float),
        where=product_time_ratio != 0,
    )

    return feature_frame
