pipeline {
    agent any
    environment {
        DOCKERFILE = 'DockerFile'
        IMAGE_NAME  = 'daki4mlops'
    }

    stages {
        stage('Checkout') {
            steps {checkout scm}
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