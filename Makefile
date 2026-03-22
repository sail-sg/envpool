SHELL          = /bin/bash
PROJECT_NAME   = envpool
PROJECT_FOLDER = $(PROJECT_NAME) third_party examples benchmark
PYTHON_FILES   = $(shell find . -type f -name "*.py")
CPP_FILES      = $(shell find $(PROJECT_NAME) -type f -name "*.h" -o -name "*.cc")
BAZEL_FILES    = $(shell find . -type f -name "*BUILD" -o -name "*.bzl")
COMMIT_HASH    = $(shell git log -1 --format=%h)
COPYRIGHT      = "Garena Online Private Limited"
BAZELOPT       =
BAZELISK_BIN   = $(shell command -v bazelisk 2>/dev/null || echo $(HOME)/go/bin/bazelisk)
BAZEL_VERSION  = 8.6.0
BAZEL          = USE_BAZEL_VERSION=$(BAZEL_VERSION) $(BAZELISK_BIN)
DATE           = $(shell date "+%Y-%m-%d")
DOCKER_TAG     = $(DATE)-$(COMMIT_HASH)
DOCKER_USER    = trinkle23897
RELEASE_PYTHON ?= $(shell python3 -c 'import sys; print("{}.{}".format(sys.version_info[0], sys.version_info[1]))')
RELEASE_SETUP_TARGET = //:setup_py$(subst .,,$(RELEASE_PYTHON))
RELEASE_ARTIFACT_LEADER ?= 3.11
CLANG_TIDY_MAJOR = 18
CLANG_TIDY_BIN = clang-tidy-$(CLANG_TIDY_MAJOR)
CLANG_TIDY_WRAPPER_DIR = $(HOME)/.cache/$(PROJECT_NAME)/bin
PATH           := $(CLANG_TIDY_WRAPPER_DIR):$(HOME)/go/bin:$(PATH)

# installation

check_install = python3 -c "import $(1)" || (cd && pip3 install $(1) --upgrade && cd -)
check_install_extra = python3 -c "import $(1)" || (cd && pip3 install $(2) --upgrade && cd -)

flake8-install:
	$(call check_install, flake8)
	$(call check_install_extra, bugbear, flake8_bugbear)

py-format-install:
	$(call check_install, isort)
	$(call check_install, yapf)

mypy-install:
	$(call check_install, mypy)

cpplint-install:
	$(call check_install, cpplint)

clang-format-install:
	command -v clang-format || sudo apt-get install -y clang-format

clang-tidy-install:
	command -v $(CLANG_TIDY_BIN) || sudo apt-get install -y $(CLANG_TIDY_BIN)
	mkdir -p $(CLANG_TIDY_WRAPPER_DIR)
	ln -sf $$(command -v $(CLANG_TIDY_BIN)) $(CLANG_TIDY_WRAPPER_DIR)/clang-tidy

go-install:
	# requires go >= 1.16
	command -v go || (sudo apt-get install -y golang-1.18 && sudo ln -sf /usr/lib/go-1.18/bin/go /usr/bin/go)

bazel-install: go-install
	command -v bazelisk || go install github.com/bazelbuild/bazelisk@latest

buildifier-install: go-install
	command -v buildifier || go install github.com/bazelbuild/buildtools/buildifier@latest

addlicense-install: go-install
	command -v addlicense || go install github.com/google/addlicense@latest

doc-install:
	$(call check_install, pydocstyle)
	$(call check_install_extra, doc8, "doc8<1")
	$(call check_install, setuptools)
	$(call check_install, pbr)
	$(call check_install, sphinx)
	$(call check_install, sphinx_rtd_theme)
	$(call check_install_extra, sphinxcontrib.spelling, sphinxcontrib.spelling pyenchant)

spelling-system-install:
	python3 -c "import ctypes.util, sys; sys.exit(0 if ctypes.util.find_library('enchant-2') or ctypes.util.find_library('enchant') else 1)" || \
	([ "$$(uname -s)" = "Linux" ] && \
	if command -v sudo >/dev/null 2>&1; then \
	  sudo apt-get update && sudo apt-get install -y libenchant-2-dev; \
	else \
	  apt-get update && apt-get install -y libenchant-2-dev; \
	fi && \
	python3 -c "import ctypes.util, sys; sys.exit(0 if ctypes.util.find_library('enchant-2') or ctypes.util.find_library('enchant') else 1)")

auditwheel-install:
	$(call check_install_extra, auditwheel, auditwheel patchelf)

release-system-install:
	if command -v dnf >/dev/null 2>&1; then \
		perl -MCompress::Zlib -e1 >/dev/null 2>&1 || \
			(dnf install -y perl-IO-Compress && dnf clean all); \
	fi

# python linter

flake8: flake8-install
	flake8 $(PYTHON_FILES) --count --show-source --statistics

py-format: py-format-install
	isort --check $(PYTHON_FILES) && yapf -r -d $(PYTHON_FILES)

mypy: mypy-install
	mypy $(PROJECT_NAME)

# c++ linter

cpplint: cpplint-install
	cpplint $(CPP_FILES)

clang-format: clang-format-install
	clang-format --style=file -i $(CPP_FILES) -n --Werror

# bazel file linter

buildifier: buildifier-install
	buildifier -r -lint=warn $(BAZEL_FILES)

# bazel build/test

bazel-pip-requirement-dev:
	cd third_party/pip_requirements && (cmp requirements.txt requirements-dev-lock.txt || ln -sf requirements-dev-lock.txt requirements.txt)

bazel-pip-requirement-release:
	cd third_party/pip_requirements && (cmp requirements.txt requirements-release-lock.txt || ln -sf requirements-release-lock.txt requirements.txt)

clang-tidy: clang-tidy-install bazel-pip-requirement-dev
	$(BAZEL) build $(BAZELOPT) //... --config=clang-tidy --config=test

bazel-debug: bazel-install bazel-pip-requirement-dev
	$(BAZEL) run $(BAZELOPT) //:setup --config=debug -- bdist_wheel
	mkdir -p dist
	cp bazel-bin/setup.runfiles/$(PROJECT_NAME)/dist/*.whl ./dist

bazel-build: bazel-install bazel-pip-requirement-dev
	$(BAZEL) run $(BAZELOPT) //:setup --config=test -- bdist_wheel
	mkdir -p dist
	cp bazel-bin/setup.runfiles/$(PROJECT_NAME)/dist/*.whl ./dist

bazel-release: bazel-install bazel-pip-requirement-release release-system-install
	$(BAZEL) run $(BAZELOPT) $(RELEASE_SETUP_TARGET) --config=release -- bdist_wheel
	mkdir -p dist
	cp bazel-bin/$(subst //:,,$(RELEASE_SETUP_TARGET)).runfiles/$(PROJECT_NAME)/dist/*.whl ./dist

bazel-test: bazel-install bazel-pip-requirement-dev
	$(BAZEL) test --test_output=all $(BAZELOPT) //... --config=test --spawn_strategy=local --color=yes

bazel-clean: bazel-install
	$(BAZEL) clean --expunge

# documentation

addlicense: addlicense-install
	addlicense -c $(COPYRIGHT) -l apache -y 2026 -check $(PROJECT_FOLDER)

docstyle: doc-install
	pydocstyle $(PROJECT_NAME) && doc8 docs && cd docs && make html SPHINXOPTS="-W"

doc: doc-install
	cd docs && make html && cd _build/html && python3 -m http.server

spelling: doc-install spelling-system-install
	cd docs && make spelling SPHINXOPTS="-W"

doc-clean:
	cd docs && make clean

doc-benchmark:
	pandoc benchmark/README.md --from markdown --to rst -s -o docs/content/benchmark.rst --columns 1000
	cd benchmark && ./plot.py --suffix png && mv *.png ../docs/_static/images/throughput

lint: buildifier flake8 py-format clang-format cpplint clang-tidy mypy docstyle spelling

format: py-format-install clang-format-install buildifier-install addlicense-install
	isort $(PYTHON_FILES)
	yapf -ir $(PYTHON_FILES)
	clang-format -style=file -i $(CPP_FILES)
	buildifier -r -lint=fix $(BAZEL_FILES)
	addlicense -c $(COPYRIGHT) -l apache -y 2026 $(PROJECT_FOLDER)

# Build docker images

docker-ci:
	docker build --network=host -t $(PROJECT_NAME):$(DOCKER_TAG) -f docker/dev.dockerfile .
	echo successfully build docker image with tag $(PROJECT_NAME):$(DOCKER_TAG)

docker-ci-push: docker-ci
	docker tag $(PROJECT_NAME):$(DOCKER_TAG) $(DOCKER_USER)/$(PROJECT_NAME):$(DOCKER_TAG)
	docker push $(DOCKER_USER)/$(PROJECT_NAME):$(DOCKER_TAG)

docker-ci-launch: docker-ci
	docker run --network=host -v /home/ubuntu:/home/github-action --shm-size=4gb -it $(PROJECT_NAME):$(DOCKER_TAG) bash

docker-dev: docker-ci
	docker run --network=host -v /:/host -v $(shell pwd):/app -v $(HOME)/.cache:/root/.cache --shm-size=4gb -it $(PROJECT_NAME):$(DOCKER_TAG) zsh

docker-release:
	docker build --network=host -t $(PROJECT_NAME)-release:$(DOCKER_TAG) -f docker/release.dockerfile .
	echo successfully build docker image with tag $(PROJECT_NAME)-release:$(DOCKER_TAG)

docker-release-push: docker-release
	docker tag $(PROJECT_NAME)-release:$(DOCKER_TAG) $(DOCKER_USER)/$(PROJECT_NAME)-release:$(DOCKER_TAG)
	docker push $(DOCKER_USER)/$(PROJECT_NAME)-release:$(DOCKER_TAG)

docker-release-launch: docker-release
	docker run --network=host -v /:/host -v $(shell pwd):/app -v $(HOME)/.cache:/root/.cache --shm-size=4gb -it $(PROJECT_NAME)-release:$(DOCKER_TAG) zsh

pypi-wheel: auditwheel-install bazel-release
	rm -rf wheelhouse
	CURRENT_WHEEL=$$(ls dist/*.whl -Art | tail -n 1); \
	python3 -m auditwheel repair --plat manylinux_2_28_x86_64 "$$CURRENT_WHEEL"; \
	if [ "$(GITHUB_ACTIONS)" = "true" ] && [ "$(RELEASE_PYTHON)" = "$(RELEASE_ARTIFACT_LEADER)" ]; then \
		for py in 3.12 3.13; do \
			if [ "$$py" = "$(RELEASE_PYTHON)" ]; then \
				continue; \
			fi; \
			$(MAKE) bazel-release RELEASE_PYTHON=$$py; \
			EXTRA_WHEEL=$$(ls dist/*.whl -Art | tail -n 1); \
			python3 -m auditwheel repair --plat manylinux_2_28_x86_64 "$$EXTRA_WHEEL"; \
			rm -f "$$EXTRA_WHEEL"; \
		done; \
	fi

release-test1:
	cd envpool && python3 make_test.py

release-test2:
	cd examples && python3 make_env.py && python3 env_step.py

release-test: release-test1 release-test2
	if [ "$(GITHUB_ACTIONS)" = "true" ] && [ "$(RELEASE_PYTHON)" != "$(RELEASE_ARTIFACT_LEADER)" ]; then \
		rm -rf wheelhouse; \
	fi
