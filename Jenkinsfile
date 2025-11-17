pipeline {
    agent any

    // Environment variables
    environment {
        GITHUB_REPO = 'tien4112004/ai-worker'
        GITHUB_USERNAME = 'tien4112004'  // GHCR
        DOCKER_REGISTRY = 'ghcr.io'
        IMAGE_NAME = "${DOCKER_REGISTRY}/${GITHUB_REPO}"
        DOCKER_COMPOSE_FILE = 'docker-compose.prod.yml'
        DEPLOY_DIR = '/opt/ai-worker'
        ENV_FILE = '/opt/ai-worker/.env.prod'
        CONTAINER_NAME = 'ai-worker-aiprimary'
    }

    options {
        buildDiscarder(logRotator(numToKeepStr: '30', daysToKeepStr: '30'))
        timestamps()
        timeout(time: 1, unit: 'HOURS')
    }

    stages {
        stage('Preparation') {
            steps {
                script {
                    echo "========== Deployment Preparation =========="
                    echo "Image: ${IMAGE_NAME}"
                    echo "Deploy Directory: ${DEPLOY_DIR}"
                    echo "Environment File: ${ENV_FILE}"
                    echo "Branch: ${env.BRANCH_NAME}"
                }
            }
        }

        stage('Validate Environment') {
            steps {
                script {
                    echo "========== Validating Environment =========="

                    // Check if Docker and Docker Compose are available
                    sh '''
                        docker --version
                        docker compose version

                        # Check if environment file exists
                        if [ ! -f "${ENV_FILE}" ]; then
                            echo "WARNING: Environment file not found at ${ENV_FILE}"
                            echo "You must create it with required environment variables:"
                            echo "- GOOGLE_API_KEY (for Gemini image generation)"
                            echo "- OPENAI_API_KEY (if using OpenAI models)"
                            echo "- DEEPSEEK_API_KEY (if using DeepSeek models)"
                            echo "- OPENROUTER_API_KEY (if using OpenRouter)"
                            echo "- Any other LLM provider API keys"
                        fi
                    '''
                }
            }
        }

        stage('Authenticate Docker Registry') {
            steps {
                script {
                    echo "========== Authenticating with GHCR =========="

                    withCredentials([string(credentialsId: 'github_pat', variable: 'GITHUB_TOKEN')]) {
                        sh '''
                            # Validate token is not empty
                            if [ -z "${GITHUB_TOKEN}" ]; then
                                echo "ERROR: GITHUB_TOKEN is empty"
                                exit 1
                            fi

                            # Set username explicitly
                            GHCR_USERNAME="tien4112004"

                            # Login to GHCR
                            echo "${GITHUB_TOKEN}" | docker login ghcr.io -u "${GHCR_USERNAME}" --password-stdin

                            echo "✓ Successfully authenticated to ghcr.io"
                        '''
                    }
                }
            }
        }

        stage('Pull Latest Image') {
            steps {
                script {
                    echo "========== Pulling Latest Docker Image =========="

                    sh '''
                        docker pull ${IMAGE_NAME}:latest || true
                        docker pull ${IMAGE_NAME}:main || true

                        # Show image info
                        docker image inspect ${IMAGE_NAME}:latest 2>/dev/null || echo "Image not found locally"
                    '''
                }
            }
        }

        stage('Stop Current Deployment') {
            steps {
                script {
                    echo "========== Stopping Current AI Worker Deployment =========="

                    sh '''
                        cd ${DEPLOY_DIR}

                        if [ -f "${DOCKER_COMPOSE_FILE}" ]; then
                            docker compose -f ${DOCKER_COMPOSE_FILE} down || true
                            echo "AI Worker service stopped via compose"
                        fi

                        # Force stop container if compose didn't work
                        if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
                            echo "Force stopping container ${CONTAINER_NAME}..."
                            docker stop ${CONTAINER_NAME} || true
                            docker rm ${CONTAINER_NAME} || true
                        fi
                    '''
                }
            }
        }

        stage('Copy Configuration') {
            steps {
                script {
                    echo "========== Copying Configuration Files =========="

                    sh '''
                        # Ensure deploy directory exists
                        mkdir -p ${DEPLOY_DIR}

                        # Copy docker-compose file to deploy directory
                        cp ${WORKSPACE}/${DOCKER_COMPOSE_FILE} ${DEPLOY_DIR}/

# TODO: Fix this
                        # Copy any additional config files if needed
                        # cp ${WORKSPACE}/secrets/* ${DEPLOY_DIR}/ 2>/dev/null || true

                        echo "Configuration copied to ${DEPLOY_DIR}"
                        echo "Files in deploy directory:"
                        ls -la ${DEPLOY_DIR}
                    '''
                }
            }
        }

        stage('Start Deployment') {
            steps {
                script {
                    echo "========== Starting AI Worker Deployment =========="

                    sh '''
                        cd ${DEPLOY_DIR}

                        # Pull latest image
                        export DOCKER_IMAGE="${IMAGE_NAME}:latest"

                        docker compose -f ${DOCKER_COMPOSE_FILE} --env-file ${ENV_FILE} pull
                        docker compose -f ${DOCKER_COMPOSE_FILE} --env-file ${ENV_FILE} up -d

                        echo "AI Worker container started successfully"

                        # Wait for service to initialize
                        echo "Waiting for AI Worker to initialize..."
                        sleep 10

                        # Show status
                        echo "========== Service Status =========="
                        docker compose -f ${DOCKER_COMPOSE_FILE} ps
                    '''
                }
            }
        }

        stage('Health Check') {
            steps {
                script {
                    echo "========== Performing Health Check =========="

                    sh '''
                        # Check if container is running
                        if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
                            echo "ERROR: Container ${CONTAINER_NAME} is not running!"
                            exit 1
                        fi

                        echo "✓ Container is running"

                        # Wait for the service to be ready
                        timeout=60
                        counter=0
                        echo "Checking API health..."

                        # Try to reach the API health endpoint
                        until curl -sf http://localhost:8083/docs > /dev/null 2>&1; do
                            counter=$((counter + 1))
                            if [ $counter -gt $timeout ]; then
                                echo "WARNING: API failed to respond within ${timeout} seconds"
                                echo "Container logs:"
                                docker logs --tail 50 ${CONTAINER_NAME}
                                break
                            fi
                            echo "Waiting for API to be ready... ($counter/$timeout)"
                            sleep 5
                        done

                        if [ $counter -le $timeout ]; then
                            echo "✓ API is responding"
                        fi

                        # Show recent logs
                        echo "========== Recent Container Logs =========="
                        docker logs --tail 20 ${CONTAINER_NAME}
                    '''
                }
            }
        }

        stage('Cleanup Old Images') {
            when {
                branch 'main'
            }
            steps {
                script {
                    echo "========== Cleaning Up Old Docker Images =========="

                    sh '''
                        # Remove dangling images
                        docker image prune -f || true

                        # Remove unused volumes
                        docker volume prune -f || true

                        # Show disk usage
                        docker system df
                    '''
                }
            }
        }
    }

    post {
        always {
            script {
                echo "========== Pipeline Completed =========="

                // Save deployment logs
                sh '''
                    mkdir -p ${WORKSPACE}/logs || true

                    # Only save logs if container exists
                    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
                        echo "Saving logs for container ${CONTAINER_NAME}..."
                        docker logs ${CONTAINER_NAME} > ${WORKSPACE}/logs/deployment.log 2>&1 || true
                    else
                        echo "Container ${CONTAINER_NAME} does not exist, skipping log collection"
                    fi

                    # Save compose status if compose file exists
                    if [ -f "${DEPLOY_DIR}/${DOCKER_COMPOSE_FILE}" ]; then
                        docker compose -f ${DEPLOY_DIR}/${DOCKER_COMPOSE_FILE} ps > ${WORKSPACE}/logs/containers.log 2>&1 || true
                    fi
                '''
            }
        }

        success {
            script {
                echo "✓ Deployment successful!"

                sh '''
                    echo "========== Deployment Summary =========="
                    echo "Service: AI Worker"
                    echo "Image: ${IMAGE_NAME}:latest"
                    echo "Container: ${CONTAINER_NAME}"
                    echo "Status:"
                    docker ps --filter "name=${CONTAINER_NAME}" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
                '''
            }
        }

        failure {
            script {
                echo "✗ Deployment failed!"

                sh '''
                    echo "========== Container Status =========="
                    docker ps -a || true

                    echo "========== Recent Logs =========="
                    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
                        docker logs --tail 100 ${CONTAINER_NAME} || true
                    else
                        echo "Container ${CONTAINER_NAME} does not exist yet"
                    fi

                    echo "========== Docker Compose Status =========="
                    if [ -f "${DEPLOY_DIR}/${DOCKER_COMPOSE_FILE}" ]; then
                        cd ${DEPLOY_DIR}
                        docker compose -f ${DOCKER_COMPOSE_FILE} ps || true
                    else
                        echo "Docker compose file not found at ${DEPLOY_DIR}"
                    fi

                    echo "========== System Resources =========="
                    docker stats --no-stream || true
                '''
            }
        }

        unstable {
            echo "⚠ Pipeline is unstable"
        }

        cleanup {
            script {
                echo "Cleaning up workspace..."
                // Uncomment if you want to clean workspace after each build
                // cleanWs()
            }
        }
    }
}
