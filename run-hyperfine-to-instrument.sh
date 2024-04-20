for f in $(cat to-instrument.txt); do
	echo "hyperfine \"python -m IPython $f/bench.ipynb\""
	echo "hyperfine \"python -m IPython $f/bench-dias.ipynb\""
	echo "hyperfine \"python -m IPython $f/bench-modin.ipynb\""
done