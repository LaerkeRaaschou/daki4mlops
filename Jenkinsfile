pipeline {
    agent any


    parameters {
        booleanParam(name: 'RUN_TRAINING', defaultValue: false, description: 'Run model training.')
        string(name: 'EPOCHS', defaultValue: '10', description: 'Number of epochs.')
    }



    environment {
        DOCKERFILE = 'DockerFile'
        IMAGE_NAME  = 'daki4mlops'
        DOCKERHUB_REPO = 'ainger24/daki4mlops'
        MLFLOW_TRACKING_URI = 'http://172.24.198.42:5050'
        MLFLOW_EXPERIMENT_NAME = 'tiny-imagenet-resnet18'
    }



    stages {
        stage('Checkout') {
            steps {
                sh 'sudo chown -R $(whoami):$(whoami) $WORKSPACE || true'
                checkout scm
            }
        }


        stage('Pull Dataset') {
            when {
                expression { params.RUN_TRAINING }
            }
            steps {
                withCredentials([
                    string(credentialsId: 'minio-user', variable: 'AWS_ACCESS_KEY_ID'),
                    string(credentialsId: 'minio-password', variable: 'AWS_SECRET_ACCESS_KEY')
                    ]) {
                        sh '''
                            set -eux
                            docker run --rm \
                                -e AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
                                -e AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
                                -v "$WORKSPACE:/repo" \
                                -w /repo \
                                dvcorg/cml:latest \
                                dvc pull data/tiny-imagenet-200.dvc
                        '''
                }
            }
        }


        stage('Docker Check') {
            steps {
                sh 'docker version'
            }
        }


        stage('Reset Docker Auth') {
            steps {
                sh '''
                    set +e
                    docker logout || true
                    rm -f ~/.docker/config.json || true
                    set -e
                '''
            }
        }


        stage('Build Docker Image') {
            steps {
                sh '''
                    set -eux
                    TAG=$(git rev-parse --short HEAD)
                    docker build -f "${DOCKERFILE}" -t "${IMAGE_NAME}:${TAG}" .
                '''
            }
        }



        stage('Unit Test in Docker') {
            steps {
                sh '''
                    set -eux
                    TAG=$(git rev-parse --short HEAD)
                    docker run --rm "${IMAGE_NAME}:${TAG}" \
                        python3 -m pytest unit_tests -v
                '''
            }
        }



        stage('Train Model') {
            when {
                expression { params.RUN_TRAINING }
            }
            steps {
                withCredentials([
                    string(credentialsId: 'wandb-api-key', variable: 'WANDB_API_KEY')
                    ]) {
                    sh '''
                        set -eux
                        TAG=$(git rev-parse --short HEAD)

                        mkdir -p "$WORKSPACE/artifacts_gr5"

                        docker run --rm --gpus all \
                            -e WANDB_API_KEY="$WANDB_API_KEY" \
                            -e MLFLOW_TRACKING_URI="$MLFLOW_TRACKING_URI" \
                            -e MLFLOW_EXPERIMENT_NAME="$MLFLOW_EXPERIMENT_NAME" \
                            -v "$WORKSPACE/data:/app/data" \
                            -v "$WORKSPACE/artifacts_gr5:/app/artifacts" \
                            "${IMAGE_NAME}:${TAG}" \
                            python train.py trainer.epochs="${EPOCHS}" compile=false
                    '''
                }
            }
        }

        stage('Evaluate Model') {
            when {expression { params.RUN_TRAINING }}
            steps {
                sh '''
                    set -eux
                    TAG=$(git rev-parse --short HEAD)

                    docker run --rm --gpus all \
                        -e MLFLOW_TRACKING_URI="$MLFLOW_TRACKING_URI" \
                        -v "$WORKSPACE/artifacts_gr5:/app/artifacts" \
                        -v "$WORKSPACE/data:/app/data" \
                        "${IMAGE_NAME}:${TAG}" \
                        python test.py compile=false
                '''
            }
        }

        stage('Register Model in MLflow') {
            when { 
                expression { params.RUN_TRAINING } 
                }
            steps {
                sh '''
                    set -eux
                    TAG=$(git rev-parse --short HEAD)

                    docker run --rm \
                    -e MLFLOW_TRACKING_URI="$MLFLOW_TRACKING_URI" \
                    -e MLFLOW_EXPERIMENT_NAME="$MLFLOW_EXPERIMENT_NAME" \
                    -v "$WORKSPACE/artifacts_gr5:/app/artifacts" \
                    "${IMAGE_NAME}:${TAG}" \
                    python register_model.py
                '''
            }
        }


        stage('Archive Model Artifacts') {
            when {
                expression { params.RUN_TRAINING }
            }
            steps {
                archiveArtifacts artifacts: 'artifacts_gr5/**', fingerprint: true, allowEmptyArchive: true
            }
        }


        stage('Push Docker container to DockerHub') {
            steps {
                withCredentials([usernamePassword(
                    credentialsId: 'dockerhub-credential',
                    usernameVariable: 'DOCKER_USER',
                    passwordVariable: 'DOCKER_PASS'
                )]) {
                    sh '''
                        set -eux
                        TAG=$(git rev-parse --short HEAD)

                        echo "$DOCKER_PASS" | docker login -u "$DOCKER_USER" --password-stdin
                        docker tag "${IMAGE_NAME}:${TAG}" "${DOCKERHUB_REPO}:${TAG}"
                        docker push "${DOCKERHUB_REPO}:${TAG}"
                        docker logout
                    '''
                }
            }
        }


        stage('Deploy') {
            when { branch 'main' }
            steps {
                sh '''
                set -eux
                TAG=$(git rev-parse --short HEAD)

                docker run --rm \
                -e MLFLOW_TRACKING_URI="$MLFLOW_TRACKING_URI" \
                -e MLFLOW_EXPERIMENT_NAME="$MLFLOW_EXPERIMENT_NAME" \
                -e GIT_COMMIT="$TAG" \
                "${IMAGE_NAME}:${TAG}" \
                python deploy.py
                '''
            }
        }
    }
}
