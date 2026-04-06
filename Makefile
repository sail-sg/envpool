SHELL          = /bin/bash
PROJECT_NAME   = envpool
PROJECT_FOLDER = $(PROJECT_NAME) third_party examples benchmark
PYTHON_FILES   = $(shell find . -type f -name "*.py")
CPP_FILES      = $(shell find $(PROJECT_NAME) -type f -name "*.h" -o -name "*.cc")
BAZEL_FILES    = $(shell find . -type f -name "*BUILD" -o -name "*.bzl")
COMMIT_HASH    = $(shell git log -1 --format=%h)
COPYRIGHT      = "Garena Online Private Limited"
BAZELOPT       =
# MSVC expects /D for preprocessor defines, so Bazel test/debug builds need an
# extra Windows-only define to expose ENVPOOL_TEST-gated alignment state.
WINDOWS_ENVPOOL_TEST_DEFINE = $(if $(filter Windows_NT,$(OS)),--cxxopt=/DENVPOOL_TEST,)
# Bazel's Windows test sandbox drops custom environment variables unless they
# are explicitly forwarded, which prevents MuJoCo render tests from seeing the
# Mesa DLL directory configured by CI/local shells.
WINDOWS_BAZEL_TEST_ENV = $(if $(filter Windows_NT,$(OS)),--test_env=ENVPOOL_DLL_DIR --test_env=MESA_GL_VERSION_OVERRIDE --test_env=GALLIUM_DRIVER --test_env=PATH,)
BAZELISK_BIN   = $(shell command -v bazelisk 2>/dev/null || echo $(HOME)/go/bin/bazelisk)
BAZEL_VERSION  = 8.6.0
BAZEL          = USE_BAZEL_VERSION=$(BAZEL_VERSION) $(BAZELISK_BIN)
BAZEL_TEST_TARGETS ?= //...
DATE           = $(shell date "+%Y-%m-%d")
DOCKER_TAG     = $(DATE)-$(COMMIT_HASH)
DOCKER_USER    = trinkle23897
RELEASE_PYTHON ?= $(shell python3 -c 'import sys; print("{}.{}".format(sys.version_info[0], sys.version_info[1]))')
RELEASE_SETUP_TARGET = //:setup_py$(subst .,,$(RELEASE_PYTHON))
PYPI_WHEEL_PLAT ?= manylinux_2_28_x86_64
WHEEL_SIZE_LIMIT_BYTES ?= 100000000
CLANG_TIDY_MAJOR = 18
CLANG_TIDY_BIN = clang-tidy-$(CLANG_TIDY_MAJOR)
CLANG_TIDY_WRAPPER_DIR = $(HOME)/.cache/$(PROJECT_NAME)/bin
PATH           := $(CLANG_TIDY_WRAPPER_DIR):$(HOME)/go/bin:$(PATH)
CLANG_TIDY_TARGET_RESOLVER = python3 scripts/clang_tidy_targets.py
ifeq ($(OS),Windows_NT)
BAZEL_RUNFILES_SUFFIX = .exe.runfiles
else
BAZEL_RUNFILES_SUFFIX = .runfiles
endif

# installation

check_install = python3 -c "import $(1)" || (cd && pip3 install $(1) --upgrade && cd -)
check_install_extra = python3 -c "import $(1)" || (cd && pip3 install $(2) --upgrade && cd -)

ruff-install:
	$(call check_install, ruff)

py-format-install:
	$(call check_install, ruff)

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

doxygen-install:
	command -v doxygen || (if command -v sudo >/dev/null 2>&1; then sudo apt-get update && sudo apt-get install -y doxygen; else apt-get update && apt-get install -y doxygen; fi)

go-install:
	# requires go >= 1.16
	command -v go || (sudo apt-get install -y golang-1.18 && sudo ln -sf /usr/lib/go-1.18/bin/go /usr/bin/go)

bazel-install: go-install
	command -v bazelisk || go install github.com/bazelbuild/bazelisk@latest

buildifier-install: go-install
	command -v buildifier || go install github.com/bazelbuild/buildtools/buildifier@latest

addlicense-install: go-install
	command -v addlicense || go install github.com/google/addlicense@latest

doc-install: doxygen-install
	$(call check_install_extra, doc8, "doc8<1")
	$(call check_install, setuptools)
	$(call check_install, pbr)
	$(call check_install, sphinx)
	$(call check_install, sphinx_rtd_theme)
	$(call check_install, breathe)

spelling-install: doc-install spelling-system-install
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

# python style / lint

ruff: ruff-install
	ruff check $(PYTHON_FILES)

py-format: py-format-install
	ruff format --check $(PYTHON_FILES)

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
	cd third_party/pip_requirements && (cmp -s requirements.txt requirements-dev-lock.txt || cp -f requirements-dev-lock.txt requirements.txt)

bazel-pip-requirement-release:
	cd third_party/pip_requirements && (cmp -s requirements.txt requirements-release-lock.txt || cp -f requirements-release-lock.txt requirements.txt)

clang-tidy: clang-tidy-install bazel-pip-requirement-dev
	targets="$${CLANG_TIDY_TARGETS:-$$($(CLANG_TIDY_TARGET_RESOLVER) | tr '\n' ' ')}"; \
	if [ -z "$$targets" ]; then \
		echo "No clang-tidy-relevant C++ changes detected; skipping."; \
		exit 0; \
	fi; \
	echo "Running clang-tidy on: $$targets"; \
	$(BAZEL) build $(BAZELOPT) $$targets --config=clang-tidy --config=test

bazel-debug: bazel-install bazel-pip-requirement-dev
	$(BAZEL) run $(BAZELOPT) //:setup --config=debug -- bdist_wheel
	mkdir -p dist
	cp bazel-bin/setup$(BAZEL_RUNFILES_SUFFIX)/$(PROJECT_NAME)/dist/*.whl ./dist

bazel-build: bazel-install bazel-pip-requirement-dev
	$(BAZEL) run $(BAZELOPT) //:setup --config=test -- bdist_wheel
	mkdir -p dist
	cp bazel-bin/setup$(BAZEL_RUNFILES_SUFFIX)/$(PROJECT_NAME)/dist/*.whl ./dist

bazel-release: bazel-install bazel-pip-requirement-release release-system-install
	$(BAZEL) run $(BAZELOPT) $(RELEASE_SETUP_TARGET) --config=release -- bdist_wheel
	mkdir -p dist
	cp bazel-bin/$(subst //:,,$(RELEASE_SETUP_TARGET))$(BAZEL_RUNFILES_SUFFIX)/$(PROJECT_NAME)/dist/*.whl ./dist

bazel-test: bazel-install bazel-pip-requirement-dev
	$(BAZEL) test --test_output=all $(BAZELOPT) $(WINDOWS_ENVPOOL_TEST_DEFINE) $(WINDOWS_BAZEL_TEST_ENV) --config=test --spawn_strategy=local --color=yes -- $(BAZEL_TEST_TARGETS)

bazel-coverage: bazel-install bazel-pip-requirement-dev
	$(BAZEL) coverage --combined_report=lcov --instrument_test_targets --test_output=errors $(BAZELOPT) $(WINDOWS_ENVPOOL_TEST_DEFINE) --config=test --spawn_strategy=local --color=yes -- $(BAZEL_TEST_TARGETS)

bazel-clean: bazel-install
	$(BAZEL) clean --expunge

# documentation

addlicense: addlicense-install
	addlicense -c $(COPYRIGHT) -l apache -y 2026 -check $(PROJECT_FOLDER)

docstyle: doc-install
	doc8 docs && cd docs && make html SPHINXOPTS="-W"

doc: doc-install
	cd docs && make html && cd _build/html && python3 -m http.server

spelling: spelling-install
	cd docs && make spelling SPHINXOPTS="-W"

doc-clean:
	cd docs && make clean

doc-benchmark:
	pandoc benchmark/README.md --from markdown --to rst -s -o docs/content/benchmark.rst --columns 1000
	cd benchmark && ./plot.py --suffix png && mv *.png ../docs/_static/images/throughput

lint: buildifier ruff py-format clang-format cpplint clang-tidy mypy docstyle spelling

format: py-format-install clang-format-install buildifier-install addlicense-install
	ruff check --fix $(PYTHON_FILES)
	ruff format $(PYTHON_FILES)
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
	python3 -m auditwheel repair --plat $(PYPI_WHEEL_PLAT) "$$CURRENT_WHEEL"
	python3 scripts/optimize_linux_wheel.py wheelhouse/*.whl
	python3 scripts/check_wheel_size.py --limit-bytes $(WHEEL_SIZE_LIMIT_BYTES) wheelhouse/*.whl

release-test1:
	tmpdir=$$(python3 -c 'import tempfile; print(tempfile.mkdtemp(prefix="envpool-release-test-"))'); \
	cd "$$tmpdir" && PYTHONPATH= python3 "$(CURDIR)/envpool/make_test.py"

release-test2:
	cd examples && python3 make_env.py && python3 env_step.py

release-test: release-test1 release-test2
