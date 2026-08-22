from __future__ import annotations

import json
import os
import time
from typing import Any, Literal
from uuid import uuid4

import pandas as pd
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from src.ml_project.config import (INFERENCE_CONTRACT_VERSION,
                                   MAX_PREDICTION_INSTANCES, MODEL_PATH,
                                   PROCESSED_METADATA_PATH)
from src.ml_project.logging import logger
from src.ml_project.modeling.predict import predict
from src.ml_project.monitoring import record_operational_metric

DEFAULT_MODEL_PATH = MODEL_PATH


def resolve_api_model_version() -> str:
    """Retorna a versao governada ou o run local associado ao artefato."""
    metadata_path = MODEL_PATH.with_name("model_metadata.json")
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        registry_info = metadata.get("model_registry") or {}
        if os.getenv("MLFLOW_USE_MODEL_REGISTRY_FOR_INFERENCE", "false").lower() in {
            "1",
            "true",
            "yes",
        } and registry_info.get("registered_model_name") and registry_info.get("version"):
            return f"{registry_info['registered_model_name']}:{registry_info['version']}"
        if metadata.get("mlflow_run_id"):
            return str(metadata["mlflow_run_id"])
    return DEFAULT_MODEL_PATH.name


class InferenceInstance(BaseModel):
    """Schema bruto aceito pela API de inferencia."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    Administrative: int = Field(ge=0)
    Administrative_Duration: float = Field(ge=0)
    Informational: int = Field(ge=0)
    Informational_Duration: float = Field(ge=0)
    ProductRelated: int = Field(ge=0)
    ProductRelated_Duration: float = Field(ge=0)
    BounceRates: float = Field(ge=0, le=1)
    ExitRates: float = Field(ge=0, le=1)
    PageValues: float = Field(ge=0)
    SpecialDay: float = Field(ge=0, le=1)
    Month: Literal["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    OperatingSystems: int = Field(ge=1)
    Browser: int = Field(ge=1)
    Region: int = Field(ge=1)
    TrafficType: int = Field(ge=1)
    VisitorType: Literal["Returning_Visitor", "New_Visitor", "Other"]
    Weekend: bool


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract_version: Literal["1.0"] = INFERENCE_CONTRACT_VERSION
    instances: list[InferenceInstance] = Field(
        min_length=1,
        max_length=MAX_PREDICTION_INSTANCES,
    )


class PredictionItem(BaseModel):
    prediction_id: str
    predicted_revenue: int


class PredictResponse(BaseModel):
    contract_version: str
    model_version: str
    predictions: list[PredictionItem]


class HealthResponse(BaseModel):
    status: str
    contract_version: str
    model_available: bool
    metadata_available: bool


def build_error_response(
    code: str,
    message: str,
    details: Any | None = None,
    http_status: int = status.HTTP_400_BAD_REQUEST,
) -> JSONResponse:
    payload: dict[str, dict[str, Any]] = {
        "error": {
            "code": code,
            "message": message,
            "details": details or [],
        }
    }
    return JSONResponse(status_code=http_status, content=payload)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Online Shoppers Inference API",
        version="0.1.0",
        description="API de inferencia para prever conversao de sessoes de e-commerce.",
    )

    @app.middleware("http")
    async def log_request(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid4()))
        started_at = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            elapsed_ms = (time.perf_counter() - started_at) * 1000
            record_operational_metric(
                "http_request",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                status_code=500,
                latency_ms=round(elapsed_ms, 3),
            )
            logger.exception(
                "REQUEST_ID={} METHOD={} PATH={} STATUS=500 LATENCY_MS={:.2f}",
                request_id,
                request.method,
                request.url.path,
                elapsed_ms,
            )
            raise

        elapsed_ms = (time.perf_counter() - started_at) * 1000
        response.headers["X-Request-ID"] = request_id
        record_operational_metric(
            "http_request",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            latency_ms=round(elapsed_ms, 3),
        )
        logger.info(
            "REQUEST_ID={} METHOD={} PATH={} STATUS={} LATENCY_MS={:.2f}",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(_request: Request, exc: RequestValidationError):
        return build_error_response(
            code="INVALID_INPUT_SCHEMA",
            message="Payload de inferencia invalido.",
            details=exc.errors(),
            http_status=422,
        )

    @app.exception_handler(FileNotFoundError)
    async def model_not_found_handler(_request: Request, exc: FileNotFoundError):
        return build_error_response(
            code="MODEL_NOT_FOUND",
            message="Artefato de modelo ou metadata nao encontrado.",
            details=[str(exc)],
            http_status=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    @app.exception_handler(ValueError)
    async def prediction_value_error_handler(_request: Request, exc: ValueError):
        message = str(exc)
        code = "MODEL_SCHEMA_MISMATCH" if "incompativel" in message else "INVALID_INPUT_SCHEMA"
        return build_error_response(
            code=code,
            message=message,
            details=[],
            http_status=status.HTTP_400_BAD_REQUEST,
        )

    @app.exception_handler(Exception)
    async def unexpected_error_handler(_request: Request, exc: Exception):
        logger.exception("Falha inesperada na API de inferencia: {}", exc)
        return build_error_response(
            code="PREDICTION_FAILED",
            message="Falha inesperada durante a inferencia.",
            details=[],
            http_status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            contract_version=INFERENCE_CONTRACT_VERSION,
            model_available=DEFAULT_MODEL_PATH.exists(),
            metadata_available=PROCESSED_METADATA_PATH.exists(),
        )

    @app.post("/predict", response_model=PredictResponse)
    def predict_endpoint(payload: PredictRequest) -> PredictResponse:
        dataframe = pd.DataFrame([instance.model_dump() for instance in payload.instances])
        predictions = predict(model_path=DEFAULT_MODEL_PATH, dataframe=dataframe)

        prediction_items = [
            PredictionItem(prediction_id=str(index), predicted_revenue=int(value))
            for index, value in enumerate(predictions.tolist())
        ]
        model_version = resolve_api_model_version()
        record_operational_metric(
            "prediction",
            instances=len(prediction_items),
            model_version=model_version,
            predicted_positive=sum(item.predicted_revenue for item in prediction_items),
        )
        logger.info(
            "Predicao concluida: contract_version={} instances={} model_version={}",
            payload.contract_version,
            len(prediction_items),
            model_version,
        )
        return PredictResponse(
            contract_version=payload.contract_version,
            model_version=model_version,
            predictions=prediction_items,
        )

    return app


app = create_app()
