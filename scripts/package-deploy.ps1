param(
    [string]$OutputPath = "dist/online-shoppers-api-deploy.zip"
)

$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$staging = Join-Path $root "dist/deploy-package"
$zipPath = Join-Path $root $OutputPath

if (Test-Path $staging) {
    Remove-Item -Recurse -Force $staging
}

New-Item -ItemType Directory -Force $staging | Out-Null
New-Item -ItemType Directory -Force (Split-Path $zipPath -Parent) | Out-Null

$requiredFiles = @(
    "Dockerfile",
    ".dockerignore",
    "pyproject.toml",
    "README.md",
    "params.yaml",
    "dvc.yaml",
    "deploy/aws/deploy-ec2.sh",
    "deploy/aws/cleanup-ec2.sh",
    "deploy/aws/README.md",
    "src/api/__init__.py",
    "src/api/api.py",
    "src/ml_project/__init__.py",
    "src/ml_project/config.py",
    "src/ml_project/dataset.py",
    "src/ml_project/features.py",
    "src/ml_project/logging.py",
    "src/ml_project/model_registry.py",
    "src/ml_project/monitoring.py",
    "src/ml_project/pipeline.py",
    "src/ml_project/preprocessing.py",
    "src/ml_project/modeling/__init__.py",
    "src/ml_project/modeling/predict.py",
    "src/ml_project/modeling/train.py",
    "configs/params.yaml",
    "data/processed/online_shoppers_metadata.json",
    "models/model.joblib",
    "models/preprocessor.joblib",
    "models/feature_names.json",
    "models/model_metadata.json",
    "models/version_info.json",
    "models/model_registry.json",
    "models/model_registry_events.json"
)

foreach ($relativePath in $requiredFiles) {
    $source = Join-Path $root $relativePath
    if (-not (Test-Path $source)) {
        throw "Arquivo obrigatorio nao encontrado: $relativePath"
    }

    $destination = Join-Path $staging $relativePath
    New-Item -ItemType Directory -Force (Split-Path $destination -Parent) | Out-Null
    Copy-Item -LiteralPath $source -Destination $destination
}

if (Test-Path $zipPath) {
    Remove-Item -Force $zipPath
}

Compress-Archive -Path (Join-Path $staging "*") -DestinationPath $zipPath -Force
Write-Host "Pacote criado em: $zipPath"
