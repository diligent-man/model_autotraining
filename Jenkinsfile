pipeline {
    agent any

    stages {
        stage('Build image') { 
            steps {
                sh "./build_container.sh"
            }
        }

        stage('Push image') { 
            steps {
                withDockerRegistry([credentialsId: Docker, url: ""])
                dockerImage.push()
            }
    }
}