#!/usr/bin/env bash
set -euo pipefail

APP_NAME="${APP_NAME:-online-shoppers-api}"

INSTANCE_IDS="$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=${APP_NAME}" "Name=instance-state-name,Values=pending,running,stopping,stopped" \
  --query 'Reservations[].Instances[].InstanceId' \
  --output text)"

if [[ -n "${INSTANCE_IDS}" ]]; then
  aws ec2 terminate-instances --instance-ids ${INSTANCE_IDS} >/dev/null
  echo "Terminated: ${INSTANCE_IDS}"
fi

aws ecr delete-repository --repository-name "${APP_NAME}" --force >/dev/null 2>&1 || true
echo "Deleted ECR repository if it existed: ${APP_NAME}"
