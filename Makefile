SHELL = /bin/bash
PROJECT_NAME = envpool
PYTHON_FILES = $(shell find ${PROJECT_NAME} -type f -name "*.py")
CPP_FILES = $(shell find ${PROJECT_NAME} -type f -name "*.h" -o -name "*.cc")

check_install = python3 -c "import $(1)" || pip3 install $(1) --upgrade
check_install_extra = python3 -c "import $(1)" || pip3 install $(2) --upgrade

# python linter

flake8:
	$(call check_install, flake8)
	$(call check_install_extra, bugbear, flake8_bugbear)
	flake8 ${PYTHON_FILES} --count --show-source --statistics

py-format:
	$(call check_install, isort)
	$(call check_install, yapf)
	isort --check ${PYTHON_FILES} && yapf -r -d ${PYTHON_FILES}

mypy:
	$(call check_install, mypy)
	mypy ${PROJECT_NAME}

# c++ linter

cpplint:
	$(call check_install, cpplint)
	cpplint ${CPP_FILES}

clang-format:
	clang-format-11 --style=Google -i ${CPP_FILES} -n --Werror

# bazel file linter

buildifier:
	buildifier -r -lint warn .

# documentation

docstyle:
	$(call check_install, pydocstyle)
	$(call check_install, doc8)
	$(call check_install, sphinx)
	$(call check_install, sphinx_rtd_theme)
	pydocstyle ${PROJECT_PATH} --convention=google && doc8 docs && cd docs && make html SPHINXOPTS="-W"

doc:
	$(call check_install, sphinx)
	$(call check_install, sphinx_rtd_theme)
	cd docs && make html && cd _build/html && python3 -m http.server

spelling:
	$(call check_install, sphinx)
	$(call check_install, sphinx_rtd_theme)
	$(call check_install_extra, sphinxcontrib.spelling, sphinxcontrib.spelling pyenchant)
	cd docs && make spelling SPHINXOPTS="-W"

clean:
	cd docs && make clean

commit-checks: buildifier flake8 py-format clang-format cpplint mypy docstyle spelling

format:
	$(call check_install, isort)
	$(call check_install, yapf)
	isort ${PYTHON_FILES}
	yapf -ir ${PYTHON_FILES}
	clang-format-11 --style=Google -i ${CPP_FILES}
	buildifier -r -lint=fix .

# Build docker images
docker:
	./scripts/build_docker.sh ${PROJECT_NAME} interactive

docker-push:
	./scripts/build_docker.sh ${PROJECT_NAME}
	./scripts/tag_and_push.sh ${PROJECT_NAME}

.PHONY: commit-checks format clean docker docker-push
