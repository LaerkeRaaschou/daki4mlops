pipeline {
    agent any
    environment {
        DOCKERFILE = 'DockerFile'
        IMAGE_NAME  = 'daki4mlops'
        DOCKERHUB_REPO = 'ainger24/daki4mlops'
    }

    stages {
        stage('Checkout') {
            steps {checkout scm}
        }

        stage('Setup Python') {
            steps {
                sh '''
                    set -eux
                    python3 --version
                    pip3 --version
                    python3 -m pip install --upgrade pip
                    if [ -f requirements.txt ]; then python3 -m pip install -r requirements.txt; fi
                    if [ -f requirements-dev.txt ]; then python3 -m pip install -r requirements-dev.txt; fi
                '''
            }
        }
        
        stage('Unit Test') {
            steps {
                sh '''
                    set -eux
                    pytest unit_tests/ -v
                '''
            }
        }
        
        stage('Docker Check') {
            steps {
                sh 'docker version'
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

        stage('Push Docker container to DockerHub') {
            steps {
                withCredentials([usernamePassword(
                    credentialsId: 'dockerhub-credentials',
                    usernameVariable: 'DOCKER_USER',
                    passwordVariable: 'DOCKER_PASS'
                )]) {
                    sh '''
                        set -eux
                        TAG=$(git rev-parse --short HEAD)

                        echo "$DOCKER_PASS" | docker login -u "$DOCKER_USER" --password-stdin
                        docker tag ${IMAGE_NAME}:${TAG} ${DOCKERHUB_REPO}:${TAG}
                        docker push ${DOCKERHUB_REPO}:${TAG}
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