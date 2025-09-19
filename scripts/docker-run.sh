#!/bin/bash

# Customer Segmentation & Recommendation Engine - Docker Deployment Script
# This script provides easy commands to build and run the analytics platform

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    print_success "Docker and Docker Compose are installed."
}

# Function to build the Docker image
build() {
    print_status "Building Customer Segmentation Analytics Docker image..."
    docker build -t customer-segmentation-analytics:latest .
    print_success "Docker image built successfully!"
}

# Function to run the complete analytics pipeline
run_analytics() {
    print_status "Starting complete analytics pipeline..."
    docker-compose up -d analytics-engine precision-analytics visualization-service
    print_success "Analytics pipeline started! Check logs with: docker-compose logs -f"
}

# Function to run only the main segmentation analysis
run_segmentation() {
    print_status "Running customer segmentation analysis..."
    docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/results:/app/results \
        customer-segmentation-analytics:latest python src/main.py
    print_success "Segmentation analysis completed!"
}

# Function to run high precision analytics
run_precision() {
    print_status "Running high precision analytics..."
    docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/results:/app/results \
        customer-segmentation-analytics:latest python high_precision_analytics.py
    print_success "Precision analytics completed!"
}

# Function to generate visualizations
run_visualizations() {
    print_status "Generating enterprise visualizations..."
    docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/results:/app/results \
        customer-segmentation-analytics:latest python create_enterprise_visualizations.py
    print_success "Visualizations generated!"
}

# Function to start Jupyter Lab
start_jupyter() {
    print_status "Starting Jupyter Lab for interactive analysis..."
    docker-compose up -d jupyter-lab
    print_success "Jupyter Lab started! Access at: http://localhost:8888"
    print_warning "Use token: analytics2024"
}

# Function to stop all services
stop() {
    print_status "Stopping all analytics services..."
    docker-compose down
    print_success "All services stopped!"
}

# Function to view logs
logs() {
    print_status "Showing analytics logs..."
    docker-compose logs -f
}

# Function to clean up Docker resources
cleanup() {
    print_status "Cleaning up Docker resources..."
    docker-compose down -v --remove-orphans
    docker system prune -f
    print_success "Cleanup completed!"
}

# Function to show status of services
status() {
    print_status "Analytics Platform Status:"
    docker-compose ps
}

# Function to show help
show_help() {
    echo "Customer Segmentation & Recommendation Engine - Docker Management"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  build           Build the Docker image"
    echo "  run             Run complete analytics pipeline"
    echo "  segmentation    Run only customer segmentation"
    echo "  precision       Run high precision analytics"
    echo "  visualize       Generate visualizations"
    echo "  jupyter         Start Jupyter Lab"
    echo "  stop            Stop all services"
    echo "  logs            View service logs"
    echo "  status          Show service status"
    echo "  cleanup         Clean up Docker resources"
    echo "  help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 build && $0 run     # Build and run complete pipeline"
    echo "  $0 segmentation        # Run segmentation only"
    echo "  $0 jupyter             # Start interactive analysis"
}

# Main script logic
case "${1:-help}" in
    "build")
        check_docker
        build
        ;;
    "run")
        check_docker
        run_analytics
        ;;
    "segmentation")
        check_docker
        run_segmentation
        ;;
    "precision")
        check_docker
        run_precision
        ;;
    "visualize")
        check_docker
        run_visualizations
        ;;
    "jupyter")
        check_docker
        start_jupyter
        ;;
    "stop")
        check_docker
        stop
        ;;
    "logs")
        check_docker
        logs
        ;;
    "status")
        check_docker
        status
        ;;
    "cleanup")
        check_docker
        cleanup
        ;;
    "help"|*)
        show_help
        ;;
esac 