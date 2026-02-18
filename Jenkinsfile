pipeline {
    agent any
    environment {
        DOCKERFILE = 'DockerFile'
        IMAGE_NAME  = 'daki4mlops'
        IMAGE_TAG   = "${env.BUILD_NUMBER}"
    }

    stages {
        stage('Docker check') {
            steps {
                sh 'docker version'
            }
        }

        stage('Build Docker Image') {
            steps {
                sh """
                    set -eux
                    docker build -f ${DOCKERFILE} -t ${IMAGE_NAME}:${IMAGE_TAG} .
                """
            }
        }
        
         stage('Unit Test (placeholder)') {
            steps {
                echo 'No unit test yet - skip.'
            }
        }
        
        stage('Test') {
            steps {
                echo 'Testing..'
            }
        }
        
        stage('Test in container') {
            steps {
                sh """
                    set -eux
                    docker run --rm ${IMAGE_NAME}:${IMAGE_TAG} python3 --version
                """
            }
        }
        
        stage('Deploy') {
            when { branch 'main' }
            steps {
                echo 'Deploying....'
            }
        }
    }
}
