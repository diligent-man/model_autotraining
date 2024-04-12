pipeline {
    agent any

    stages {
        stage('Build & Push image') { 
            steps {                
                withDockerRegistry([credentialsId: "348feac3-bb81-48e3-a46d-3003d1d5d243", url: ""]){    
                    sh "bash build_container.sh"
                }
            }
        }
    }
}