from datetime import datetime, timedelta

from airflow import DAG

from airflow.utils.dates import days_ago
from airflow_operator.my_operator import CustomPythonOperator

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email": ["airflow@example.com"],
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "start_date": datetime(2021, 8, 6),
}
with DAG(
    "L2R_DAG",
    default_args=default_args,
    description="My ETL and L2R dag with CustomPythonOperator",
    schedule_interval=timedelta(days=2),
    start_date=days_ago(2),
    tags=["DIPLOMA"],
    concurrency=2,
    max_active_runs=2,
) as dag:
    configs_path = "/home/huvi/recom/configs/ETL/"
    w2vec_config = "w2vec.yaml"
    etl_config = "preprocess.yaml"
    get_candidates_cfg = "get_candidates.yaml"
    concat_tables_first_cfg = "concat_tables_first.yaml"
    concat_tables_second_cfg = "concat_tables_second.yaml"
    get_categories_cfg = "get_categories.yaml"
    explode_table_cfg = "explode.yaml"
    prob_features_cfg = "generate.yaml"
    
    # ETL
    preprocess = CustomPythonOperator(
        task_id="preprocess",
        config_path=configs_path + etl_config
    )
    w2vec = CustomPythonOperator(
        task_id="train_w2vec",
        config_path=configs_path + w2vec_config
    )
    get_categories = CustomPythonOperator(
        task_id="get_categories",
        config_path=configs_path + get_categories_cfg
    )
    get_candidates = CustomPythonOperator(
        task_id="get_candidates",
        config_path=configs_path + get_candidates_cfg
    )
    concat_tables_first = CustomPythonOperator(
        task_id="concat_tables_first",
        config_path=configs_path + concat_tables_first_cfg
    )
    concat_tables_second = CustomPythonOperator(
        task_id="concat_tables_second",
        config_path=configs_path + concat_tables_second_cfg
    )
    explode_tables = CustomPythonOperator(
        task_id="explode",
        config_path=configs_path + explode_table_cfg
    )
    prob_features = CustomPythonOperator(
        task_id="prob_features",
        config_path=configs_path + prob_features_cfg
    )

    # Models Training
    configs_path = "/home/huvi/recom/configs/L2R/"
    catboost_train_cfg = "train_catboost.yaml"
    l2r_train_cfg = "train_lambda.yaml"
    evaluate_cfg = "evaluate.yaml"
    catboost_train = CustomPythonOperator(
        task_id="catboost_train",
        config_path=configs_path + catboost_train_cfg
    )
    l2r_train = CustomPythonOperator(
        task_id="l2r_train",
        config_path=configs_path + l2r_train_cfg
    )
    evaluate = CustomPythonOperator(
        task_id="evaluate_on_test",
        config_path=configs_path + evaluate_cfg
    )

    get_candidates >> concat_tables_first
    concat_tables_first >> explode_tables
    explode_tables >> prob_features
    prob_features >> concat_tables_second
    concat_tables_second >> catboost_train 
    catboost_train >> l2r_train
    l2r_train >> evaluate
    preprocess >> [w2vec, get_categories] >> get_candidates
