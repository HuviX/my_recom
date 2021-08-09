CODE = .
pretty:
	black --target-version py38 --line-length 79 $(CODE)
	isort **/*.py