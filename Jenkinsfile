pipeline {
    agent any

    stages {
        stage('Build & Push image') { 
            steps {
                withDockerRegistry([credentialsId: Docker, url: ""]){
                    sh "./build_container.sh"
                }
            }
        }
    }
}