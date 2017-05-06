#!/bin/sh

for i in `seq 0 14`
do
	echo "-----" >> c-result
	./exec_multitest -n $i -c >> c-result
done
