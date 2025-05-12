from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.bash import BashOperator
from datetime import datetime
import uuid

default_args = {
    "start_date": datetime(2025, 5, 10),
    "catchup": False
}

with DAG("scrape_labeled_articles",
         default_args=default_args,
         schedule_interval="@daily",
         start_date=datetime(2025, 5, 10),
         catchup=False,
         description="Scrape and clean Urdu news articles") as dag:
    
    run_scraper = DockerOperator(
        task_id="run_scraper_labeled",
        image="scraper_labeled:latest",
        container_name=f"scraper_{uuid.uuid4().hex[:8]}",
        api_version="auto",
        auto_remove=True,
        docker_url="unix:///var/run/docker.sock",
        network_mode="scraper_network",
        mount_tmp_dir=False,
        environment={
            "POSTGRES_HOST": "postgres",
            "POSTGRES_DB": "news_db",
            "POSTGRES_USER": "affan",
            "POSTGRES_PASSWORD": "pass123"
        },
        command="python scrapper.py"
    )

    run_cleaner = DockerOperator(
        task_id="run_cleaner",
        image="scraper_labeled:latest",
        container_name=f"cleaner_{uuid.uuid4().hex[:8]}",
        api_version="auto",
        auto_remove=True,
        docker_url="unix:///var/run/docker.sock",
        network_mode="scraper_network",
        mount_tmp_dir=False,
        environment={
            "POSTGRES_HOST": "postgres",
            "POSTGRES_DB": "news_db",
            "POSTGRES_USER": "affan",
            "POSTGRES_PASSWORD": "pass123"
        },
        command="python cleaner.py"
    )

    
    run_ml_training = DockerOperator(
        task_id="run_ml_training",
        image="ml_service:latest",
        container_name=f"ml_trainer_{uuid.uuid4().hex[:8]}",
        api_version="auto",
        auto_remove=True,
        docker_url="unix:///var/run/docker.sock",
        network_mode="scraper_network",
        mount_tmp_dir=False,
        environment={
            "POSTGRES_HOST": "postgres",
            "POSTGRES_DB": "news_db",
            "POSTGRES_USER": "affan",
            "POSTGRES_PASSWORD": "pass123",
            "MLFLOW_TRACKING_URI": "http://mlflow:5000"
        },
        command="python train_model.py"
    )

    restart_streamlit = BashOperator(
    task_id="restart_streamlit",
    bash_command="docker restart streamlit && echo 'Streamlit restarted!'"
    )

    run_scraper >> run_cleaner >> run_ml_training >> restart_streamlit
