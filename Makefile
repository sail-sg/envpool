SHELL        = /bin/bash
PROJECT_NAME = envpool
PYTHON_FILES = $(shell find . -type f -name "*.py")
CPP_FILES    = $(shell find $(PROJECT_NAME) -type f -name "*.h" -o -name "*.cc")
COMMIT_HASH  = $(shell git log -1 --format=%h)
COPYRIGHT    = "Garena Online Private Limited"
BAZELOPT     =
PATH         := $(HOME)/go/bin:$(PATH)

# installation

check_install = python3 -c "import $(1)" || pip3 install $(1) --upgrade
check_install_extra = python3 -c "import $(1)" || pip3 install $(2) --upgrade

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
	command -v clang-format-11 || sudo apt-get install -y clang-format-11

go-install:
	# requires go >= 1.16
	command -v go || (sudo apt-get install -y golang-1.16 && sudo ln -sf /usr/lib/go-1.16/bin/go /usr/bin/go)

bazel-install: go-install
	command -v bazel || (go install github.com/bazelbuild/bazelisk@latest && ln -sf $(HOME)/go/bin/bazelisk $(HOME)/go/bin/bazel)

buildifier-install: go-install
	command -v buildifier || go install github.com/bazelbuild/buildtools/buildifier@latest

addlicense-install: go-install
	command -v addlicense || go install github.com/google/addlicense@latest

doc-install:
	$(call check_install, pydocstyle)
	$(call check_install, doc8)
	$(call check_install, sphinx)
	$(call check_install, sphinx_rtd_theme)
	$(call check_install_extra, sphinxcontrib.spelling, sphinxcontrib.spelling pyenchant)

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
	clang-format-11 --style=file -i $(CPP_FILES) -n --Werror

# bazel file linter

buildifier: buildifier-install
	buildifier -r -lint=warn .

# bazel build/test

bazel-build: bazel-install
	bazel build $(BAZELOPT) //... --config=release
	bazel run $(BAZELOPT) //:setup --config=release -- bdist_wheel
	mkdir -p dist
	cp bazel-bin/setup.runfiles/$(PROJECT_NAME)/dist/*.whl ./dist

bazel-test: bazel-install
	bazel test --test_output=all $(BAZELOPT) //... --config=release

bazel-clean:
	rm -rf $(HOME)/.cache/bazel

# documentation

addlicense: addlicense-install
	addlicense -c $(COPYRIGHT) -l apache -y 2021 -check $(PROJECT_NAME) third_party

docstyle: doc-install
	pydocstyle $(PROJECT_NAME) && doc8 docs && cd docs && make html SPHINXOPTS="-W"

doc: doc-install
	cd docs && make html && cd _build/html && python3 -m http.server

spelling: doc-install
	cd docs && make spelling SPHINXOPTS="-W"

doc-clean:
	cd docs && make clean

lint: buildifier flake8 py-format clang-format cpplint mypy docstyle spelling

format: py-format-install clang-format-install buildifier-install addlicense-install
	isort $(PYTHON_FILES)
	yapf -ir $(PYTHON_FILES)
	clang-format-11 -style=file -i $(CPP_FILES)
	buildifier -r -lint=fix .
	addlicense -c $(COPYRIGHT) -l apache -y 2021 $(PROJECT_NAME) third_party

# Build docker images

docker:
	docker build -t $(PROJECT_NAME):$(COMMIT_HASH) -f Dockerfile .
	docker run --mount type=bind,source=/,target=/host -it $(PROJECT_NAME):$(COMMIT_HASH) bash
	echo successfully build docker image with tag $(PROJECT_NAME):$(COMMIT_HASH)
