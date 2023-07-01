SCRIPTS := $(CURDIR)/scripts
SRC     := $(CURDIR)/alpha_zero

PYTEST_ADDOPTS := --cov=$(SRC) --cov-report=term --cov-report=term-missing --cov-report=xml

all:

clean:
	@ find $(CURDIR) -type d -name '__pycache__'   -exec $(RM) --recursive --verbose '{}' +
	@ find $(CURDIR) -type d -name '.pytest_cache' -exec $(RM) --recursive --verbose '{}' +
	@ find $(CURDIR) -type f -name '.coverage*'    -exec $(RM) --verbose '{}' +
	@ find $(CURDIR) -type f -name 'coverage.xml'  -exec $(RM) --verbose '{}' +

pretty: black prettier

setup: $(SCRIPTS)/install-torch.sh
	poetry install
	bash $(SCRIPTS)/install-torch.sh

test:
	pytest $(PYTEST_ADDOPTS)

#####################
# Auxiliary targets #
#####################

black:
	isort --profile black $(CURDIR)
	black $(CURDIR)

prettier: $(CURDIR)/.gitignore
	prettier --write --ignore-path=$< $(CURDIR)
