pipeline {
    agent any

    options {
        skipDefaultCheckout(true)

    }

    triggers{
        pollSCM('H/5 * * * *')
    }


    parameters {
        booleanParam(name: 'RUN_TRAINING', defaultValue: false, description: 'Run model training.')
        string(name: 'EPOCHS', defaultValue: '50', description: 'Number of epochs.')
        string(name: 'OPTIMIZER', defaultValue: 'adamw', description: 'Optimizer to use for training.')
        string(name: 'LEARNING_RATE', defaultValue: '0.001', description: 'Learning rate for training.')
        string(name: 'WEIGHT_DECAY', defaultValue: '0.01', description: 'Weight decay for training.')
        string(name: 'BATCH_SIZE', defaultValue: '64', description: 'Batch size for training.')
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
                checkout scm
            }
        }

        stage('Pull Dataset') {
            when { expression { params.RUN_TRAINING } }
            steps {
                withCredentials([
                    string(credentialsId: 'minio-user', variable: 'AWS_ACCESS_KEY_ID'),
                    string(credentialsId: 'minio-password', variable: 'AWS_SECRET_ACCESS_KEY')
                ]) {
                    sh '''
                        set -eux
                        HOST_UID=$(id -u)
                        HOST_GID=$(id -g)
                        docker run --rm \
                            -e AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
                            -e AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
                            -e HOST_UID="$HOST_UID" \
                            -e HOST_GID="$HOST_GID" \
                            -v "$WORKSPACE:/repo" \
                            -w /repo \
                            dvcorg/cml:latest \
                            sh -c 'dvc pull data/tiny-imagenet-200.dvc && chown -R $HOST_UID:$HOST_GID /repo && chmod -R u+rwX /repo'
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
                    python3 -m pytest unit_tests -v \
                        --cov=train \
                        --cov=test \
                        --cov=inference \
                        --cov=drift_detection \
                        --cov=runtime_metrics \
                        --cov=monitoring_api \
                        --cov=publish_monitoring_metrics \
                        --cov=data \
                        --cov=model \
                        --cov-report=term
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
                            python train.py \
                                trainer.epochs="${EPOCHS}" \
                                optimizer="${OPTIMIZER}" \
                                optimizer.lr="${LEARNING_RATE}" \
                                data.batch_size="${BATCH_SIZE}" \
                                optimizer.weight_decay="${WEIGHT_DECAY}" \
                                compile=false

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


        stage('Generate Model Card') {
            when { expression { params.RUN_TRAINING } }
            steps {
                sh '''
                    set -eux
                    TAG=$(git rev-parse --short HEAD)

                    docker run --rm \
                        -e MLFLOW_TRACKING_URI="$MLFLOW_TRACKING_URI" \
                        -e MLFLOW_EXPERIMENT_NAME="$MLFLOW_EXPERIMENT_NAME" \
                        -v "$WORKSPACE/artifacts_gr5:/app/artifacts" \
                        "${IMAGE_NAME}:${TAG}" \
                        python model/generate_model_card.py
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


        stage('Deployment Pipeline') {
            when {
                allOf {
                    branch 'main'
                    expression { params.RUN_TRAINING }
                }
            }
            steps {
                sh '''
                set -eux
                TAG=$(git rev-parse --short HEAD)
                API_CONTAINER="daki-monitoring-api-${TAG}"
                NETWORK="daki-deployment-net-${TAG}"

                cleanup_deployment() {
                    docker rm -f "$API_CONTAINER" >/dev/null 2>&1 || true
                    docker network rm "$NETWORK" >/dev/null 2>&1 || true
                }
                trap cleanup_deployment EXIT

                cleanup_deployment
                docker network create "$NETWORK"

                docker run -d \
                    --name "$API_CONTAINER" \
                    --network "$NETWORK" \
                    --network-alias monitoring-api \
                    "${IMAGE_NAME}:${TAG}" \
                    uvicorn monitoring_api:app --host 0.0.0.0 --port 8000

                sleep 5

                mkdir -p "$WORKSPACE/results"

                docker run --rm --gpus all \
                    --network "$NETWORK" \
                    -e MLFLOW_TRACKING_URI="$MLFLOW_TRACKING_URI" \
                    -e MLFLOW_EXPERIMENT_NAME="$MLFLOW_EXPERIMENT_NAME" \
                    -e GIT_COMMIT="$TAG" \
                    -v "$WORKSPACE/data:/app/data" \
                    -v "$WORKSPACE/artifacts_gr5:/app/artifacts" \
                    -v "$WORKSPACE/results:/app/results" \
                    "${IMAGE_NAME}:${TAG}" \
                    sh -c '
                        python inference.py carbontracker=true &&
                        python drift_detection.py &&
                        python publish_monitoring_metrics.py \
                            --monitoring-url http://monitoring-api:8000 \
                            --runtime-metrics-path /app/results/runtime_metrics.jsonl \
                            --drift-summary-path results/drift/drift_summary.json &&
                        python deploy_model.py
                    '
                '''
            }
        }
    }
        post {
            always {
                sh '''
                    set +e
                    HOST_UID=$(id -u)
                    HOST_GID=$(id -g)
                    docker run --rm \
                        -e HOST_UID="$HOST_UID" \
                        -e HOST_GID="$HOST_GID" \
                        -v "$WORKSPACE:/repo" \
                        alpine:latest \
                        sh -c 'chown -R $HOST_UID:$HOST_GID /repo && chmod -R u+rwX /repo' || true

                    rm -rf "$WORKSPACE/.dvc/cache" 2>/dev/null || true
                    rm -rf "$WORKSPACE/data" 2>/dev/null || true
                    rm -rf "$WORKSPACE/artifacts_gr5" 2>/dev/null || true

                    TAG=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)
                    docker ps -aq --filter ancestor=${IMAGE_NAME}:${TAG} | xargs -r docker rm -f || true
                    docker ps -aq --filter ancestor=${DOCKERHUB_REPO}:${TAG} | xargs -r docker rm -f || true
                    docker rmi ${IMAGE_NAME}:${TAG} >/dev/null 2>&1 || true
                    docker rmi ${DOCKERHUB_REPO}:${TAG} >/dev/null 2>&1 || true
                    docker image prune -f >/dev/null 2>&1 || true
                    docker builder prune -f >/dev/null 2>&1 || true
                '''
                cleanWs(deleteDirs: true, disableDeferredWipeout: true)
            }
        }
}
