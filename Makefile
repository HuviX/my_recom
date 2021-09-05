CODE = .
pretty:
	black --target-version py38 --line-length 79 $(CODE)
	isort **/*.py
env:
	python -m pip install -e models_module/
	python -m pip install -e ETL_module/
	python -m pip install -e L2R_module/
	python -m pip install -e airflow_module/
