#!/bin/bash

lux=/home/swyngaard/src/lux-v08-x86_64-sse2-NoOpenCL/luxconsole

cmd=

for file in output/frame????.lxs
do
	$lux $file
done;



