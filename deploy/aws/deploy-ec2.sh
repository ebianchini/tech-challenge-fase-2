#!/usr/bin/env bash
set -euo pipefail

APP_NAME="${APP_NAME:-online-shoppers-api}"
INSTANCE_TYPE="${INSTANCE_TYPE:-t3.micro}"
AWS_REGION="${AWS_REGION:-$(aws configure get region)}"
AWS_ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${APP_NAME}:latest"
SG_NAME="${APP_NAME}-sg"

echo "Building and pushing ${ECR_URI}"
aws ecr describe-repositories --repository-names "${APP_NAME}" >/dev/null 2>&1 \
  || aws ecr create-repository --repository-name "${APP_NAME}" >/dev/null

aws ecr get-login-password --region "${AWS_REGION}" \
  | docker login --username AWS --password-stdin "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

docker build -t "${APP_NAME}:latest" .
docker tag "${APP_NAME}:latest" "${ECR_URI}"
docker push "${ECR_URI}"

VPC_ID="$(aws ec2 describe-vpcs --query 'Vpcs[0].VpcId' --output text)"
SUBNET_ID="$(aws ec2 describe-subnets \
  --filters "Name=vpc-id,Values=${VPC_ID}" \
  --query 'Subnets[0].SubnetId' \
  --output text)"

IGW_ID="$(aws ec2 describe-internet-gateways \
  --filters "Name=attachment.vpc-id,Values=${VPC_ID}" \
  --query 'InternetGateways[0].InternetGatewayId' \
  --output text)"

if [[ "${IGW_ID}" == "None" ]]; then
  IGW_ID="$(aws ec2 create-internet-gateway \
    --query 'InternetGateway.InternetGatewayId' \
    --output text)"
  aws ec2 attach-internet-gateway --internet-gateway-id "${IGW_ID}" --vpc-id "${VPC_ID}"
fi

ROUTE_TABLE_ID="$(aws ec2 describe-route-tables \
  --filters "Name=vpc-id,Values=${VPC_ID}" "Name=association.main,Values=true" \
  --query 'RouteTables[0].RouteTableId' \
  --output text)"

aws ec2 create-route \
  --route-table-id "${ROUTE_TABLE_ID}" \
  --destination-cidr-block 0.0.0.0/0 \
  --gateway-id "${IGW_ID}" >/dev/null 2>&1 || true

aws ec2 modify-subnet-attribute \
  --subnet-id "${SUBNET_ID}" \
  --map-public-ip-on-launch

echo "Using subnet ${SUBNET_ID} with Internet Gateway ${IGW_ID}"

SG_ID="$(aws ec2 describe-security-groups \
  --filters "Name=group-name,Values=${SG_NAME}" "Name=vpc-id,Values=${VPC_ID}" \
  --query 'SecurityGroups[0].GroupId' \
  --output text)"

if [[ "${SG_ID}" == "None" ]]; then
  SG_ID="$(aws ec2 create-security-group \
    --group-name "${SG_NAME}" \
    --description "Online Shoppers API access" \
    --vpc-id "${VPC_ID}" \
    --query 'GroupId' \
    --output text)"
fi

aws ec2 authorize-security-group-ingress \
  --group-id "${SG_ID}" \
  --protocol tcp \
  --port 8000 \
  --cidr 0.0.0.0/0 >/dev/null 2>&1 || true

AMI_ID="$(aws ec2 describe-images \
  --owners amazon \
  --filters 'Name=name,Values=al2023-ami-2023*-x86_64' 'Name=state,Values=available' \
  --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
  --output text)"

USER_DATA_FILE="$(mktemp)"
cat > "${USER_DATA_FILE}" <<EOF
#!/usr/bin/env bash
set -euxo pipefail

dnf install -y docker
systemctl enable --now docker

aws ecr get-login-password --region "${AWS_REGION}" \
  | docker login --username AWS --password-stdin "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

docker rm -f "${APP_NAME}" || true
docker pull "${ECR_URI}"
docker run -d --name "${APP_NAME}" --restart unless-stopped -p 8000:8000 "${ECR_URI}"
EOF

INSTANCE_ID="$(aws ec2 run-instances \
  --image-id "${AMI_ID}" \
  --instance-type "${INSTANCE_TYPE}" \
  --subnet-id "${SUBNET_ID}" \
  --associate-public-ip-address \
  --security-group-ids "${SG_ID}" \
  --iam-instance-profile Name=LabInstanceProfile \
  --user-data "file://${USER_DATA_FILE}" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${APP_NAME}}]" \
  --query 'Instances[0].InstanceId' \
  --output text)"

rm -f "${USER_DATA_FILE}"

aws ec2 wait instance-running --instance-ids "${INSTANCE_ID}"

PUBLIC_IP="$(aws ec2 describe-instances \
  --instance-ids "${INSTANCE_ID}" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text)"

echo
echo "Instance: ${INSTANCE_ID}"
echo "Health:   http://${PUBLIC_IP}:8000/health"
echo "API:      http://${PUBLIC_IP}:8000"

echo
echo "Waiting for the API health check..."
for attempt in {1..30}; do
  if curl -fsS "http://${PUBLIC_IP}:8000/health" >/dev/null; then
    echo "API is ready: http://${PUBLIC_IP}:8000/health"
    exit 0
  fi
  sleep 10
done

echo "API did not become ready in time. Check cloud-init logs with:" >&2
echo "aws ec2 get-console-output --instance-id ${INSTANCE_ID} --latest --query Output --output text" >&2
exit 1
