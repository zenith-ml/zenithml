all: test

lint: FORCE
	flake8
	black --check .
	isort --check .

format: license FORCE
	black .
	isort .

install: FORCE
	pip install -e .[dev,doc,tf,torch]

doctest: FORCE
	$(MAKE) -C docs doctest

test: lint FORCE
	pytest -v test

clean: FORCE
	git clean -dfx -e numpyro.egg-info

docs: FORCE
	$(MAKE) -C docs html

FORCE: