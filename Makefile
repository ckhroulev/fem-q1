build: Q12D.pyx Q13D.pyx
	python setup.py build_ext --inplace

clean:
	rm -f *.pyc *.so Q12D.c Q13D.c
	python setup.py clean --all

install:
	python setup.py install --user
