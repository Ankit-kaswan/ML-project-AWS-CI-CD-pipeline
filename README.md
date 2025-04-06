# End to End ML project 
📌 Overview

This project demonstrates a complete end-to-end machine learning pipeline using Flask for deployment. It includes data ingestion, preprocessing, model training, evaluation, and deployment.

🚀 Features

Data Ingestion: Reads data from CSV/DB.

Data Preprocessing: Cleans and transforms data.

Model Training: Uses multiple ML algorithms.

Model Evaluation: Selects the best model.

Model Deployment: Exposes the model as an API.

CI/CD Integration: Automated testing & deployment.


###  Full Docker Setup and EC2 Runner Configuration (Ubuntu EC2)
```shell
# [Optional] Update & Upgrade packages
sudo apt-get update -y
sudo apt-get upgrade -y

# [Required] Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add current user (ubuntu) to docker group
sudo usermod -aG docker ubuntu

# Refresh group (so you don't need to log out/log in)
newgrp docker

# [Optional] Verify Docker works
docker --version
docker run hello-world

```
### GitHub Self-hosted Runner Setup
 Do this inside your EC2 instance inside your project folder or a directory you create for the runner.
 
```shell
# Create a directory for the runner
mkdir actions-runner && cd actions-runner

# Download the latest runner package (for Ubuntu x64)
curl -o actions-runner-linux-x64-2.316.0.tar.gz -L https://github.com/actions/runner/releases/download/v2.316.0/actions-runner-linux-x64-2.316.0.tar.gz

# Extract it
tar xzf actions-runner-linux-x64-2.316.0.tar.gz

# Configure the runner (replace values accordingly)
./config.sh --url https://github.com/<your-username-or-org>/<repo-name> \
            --token <your-registration-token>

# Start the runner
./run.sh

```
###  GitHub Secrets to Add for Docker + ECR Push (GitHub > Repo > Settings > Secrets and variables > Actions)
```
AWS_ACCESS_KEY_ID	# Your AWS access key
AWS_SECRET_ACCESS_KEY #	Your AWS secret access key
AWS_REGION	# us-east-1
AWS_ECR_LOGIN_URI #	566373416292.dkr.ecr.ap-south-1.amazonaws.com
ECR_REPOSITORY_NAME #	simple-app
```