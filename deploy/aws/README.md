# Deploy no AWS Academy Learner Lab

Fluxo usado para publicar a API no Learner Lab com CloudShell, ECR, EC2 e Docker.

## Gerar pacote

No Windows:

```powershell
.\scripts\package-deploy.ps1
```

Envie `dist/online-shoppers-api-deploy.zip` para o CloudShell pelo menu **Actions > Upload file**.

## Publicar

No CloudShell:

```bash
rm -rf ~/online-shoppers-api
unzip online-shoppers-api-deploy.zip -d online-shoppers-api
cd online-shoppers-api
chmod +x deploy/aws/deploy-ec2.sh
deploy/aws/deploy-ec2.sh
```

O script cria/atualiza o ECR, faz build/push da imagem, prepara rede e Security Group, sobe uma EC2 e espera o `/health`.

## Testar

```bash
curl http://IP_PUBLICO:8000/health
```

```bash
curl -X POST http://IP_PUBLICO:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"contract_version":"1.0","instances":[{"Administrative":0,"Administrative_Duration":0.0,"Informational":0,"Informational_Duration":0.0,"ProductRelated":1,"ProductRelated_Duration":0.0,"BounceRates":0.2,"ExitRates":0.2,"PageValues":0.0,"SpecialDay":0.0,"Month":"Feb","OperatingSystems":1,"Browser":1,"Region":1,"TrafficType":1,"VisitorType":"Returning_Visitor","Weekend":false}]}'
```

## Limpar recursos

```bash
chmod +x deploy/aws/cleanup-ec2.sh
deploy/aws/cleanup-ec2.sh
```
