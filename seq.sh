#!/bin/sh

for i in `seq 0 14`
do
	./exec_multitest -n $i -c >> c-result
	./exec_multitest -n $i -m >> m-result
done
