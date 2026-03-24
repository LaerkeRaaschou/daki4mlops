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
    }


    stages {
        stage('Checkout') {
            steps {checkout scm}
        }

        stage('Pull Dataset') {
            when {
                expression { params.RUN_TRAINING }
            }
            steps {
                sh '''
                    set -eux
                    docker run --rm \
                    -v "$WORKSPACE:/repo" \
                    -w /repo \
                    iterativeai/dvc \
                    dvc pull data/tiny-imagenet-200.dvc
                    test -d data/tiny-imagenet-200/train
                '''
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
                withCredentials([string(credentialsId: 'wandb-api-key', variable: 'WANDB_API_KEY')]) {
                    sh '''
                        set -eux
                        TAG=$(git rev-parse --short HEAD)

                        docker run --rm \
                            --gpus all \
                            -e WANDB_API_KEY="$WANDB_API_KEY" \
                            -v "$WORKSPACE/data:/app/data" \
                            "${IMAGE_NAME}:${TAG}" \
                            python train.py trainer.epochs="${EPOCHS}" compile=false
                    '''
                }
            }
        }

        stage('Archive Model Artifacts') {
            when {
                expression { params.RUN_TRAINING }
            }
            steps {
                echo 'Model artifact archiving will be added next.'
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

        
        stage('Test in container') {
            steps {
                sh '''
                    set -eux
                    TAG=$(git rev-parse --short HEAD)
                    docker run --rm "${IMAGE_NAME}:${TAG}" python3 --version
                '''
            }
        }
        

        stage('Deploy') {
            when { branch 'main' }
            steps {
                sh '''
                set -eux
                TAG=$(git rev-parse --short HEAD)
                '''
            }
        }
    }
}