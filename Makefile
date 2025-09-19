# Makefile for Customer Segmentation & Business Intelligence Platform

.PHONY: help install install-dev test lint format clean run-pipeline run-dashboard docker-build docker-run

help:
	@echo "Available commands:"
	@echo "  install      Install package and dependencies"
	@echo "  install-dev  Install package with development dependencies"
	@echo "  test         Run unit tests"
	@echo "  lint         Run code linting"
	@echo "  format       Format code with black"
	@echo "  clean        Clean up temporary files"
	@echo "  run-pipeline Run main business intelligence pipeline"
	@echo "  run-dashboard Run executive dashboard"
	@echo "  docker-build Build Docker image"
	@echo "  docker-run   Run Docker container"

install:
	pip install -r requirements/requirements.txt
	pip install -e .

install-dev:
	pip install -r requirements/requirements.txt
	pip install -r requirements/requirements-dev.txt
	pip install -e .

test:
	python -m pytest tests/ -v

lint:
	flake8 src/
	black --check src/

format:
	black src/
	isort src/

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/

run-pipeline:
	python src/main.py

run-dashboard:
	streamlit run src/visualization/executive_dashboard.py

docker-build:
	docker build -f config/Dockerfile -t customer-segmentation-bi .

docker-run:
	docker-compose -f config/docker-compose.yml up
