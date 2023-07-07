RUN     := $(CURDIR)/run
SCRIPTS := $(CURDIR)/scripts
SRC     := $(CURDIR)/alpha_zero

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

JOB_ID    != date +%Y-%m-%dT%H-%M-%S
WORKSPACE := $(RUN)/$(JOB_ID)
run: $(WORKSPACE)/archive.tar.gz
	tar --extract --file=$< --gzip --directory=$(WORKSPACE)
	tmux new-session -d -c $(WORKSPACE) -s $(JOB_ID) "$(MAKE) train"
	tmux pipe-pane -o -t $(JOB_ID) "cat > $(WORKSPACE)/run.log"

PYTEST_ADDOPTS := --cov=$(SRC) --cov-report=term --cov-report=term-missing --cov-report=xml
test:
	pytest $(PYTEST_ADDOPTS)

train: $(CURDIR)/train.py
	python $<

#####################
# Auxiliary targets #
#####################

$(WORKSPACE):
	@ mkdir --parents --verbose $@

$(WORKSPACE)/archive.tar.gz: | $(WORKSPACE)
	git ls-files | tar --create --file=$@ --gzip --files-from="-"

black:
	isort --profile black $(CURDIR)
	black $(CURDIR)

prettier: $(CURDIR)/.gitignore
	prettier --write --ignore-path=$< $(CURDIR)
