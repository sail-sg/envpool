pyenv install 3.7.12 &
pid37=$!
pyenv install 3.8.12 &
pid38=$!
pyenv install 3.9.8 &
pid39=$!
wait $pid37 $pid38 $pid39
