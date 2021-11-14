# release test
for ver in 3.7.12 3.8.12 3.9.8
do
	pyenv global $ver
	i=`echo $ver | cut -d "." -f 2`
	pip install `ls wheelhouse/*cp3$i*`
	make release-test
done
