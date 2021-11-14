BAZELOPT="--remote_cache=http://bazel-cache-http.ai.seacloud.garenanow.com"

# compile each evrsion
for ver in 3.7.12 3.8.12 3.9.8
do
	pyenv global $ver
	make bazel-build BAZELOPT=$BAZELOPT
done

# install auditwheel
pip install auditwheel
for i in `ls dist`
do
       auditwheel repair --plat manylinux_2_17_x86_64 dist/$i
done
