#!/bin/bash

USER=pi
PASSWORD=raspberry

name_worker=(raspi8 raspi9 raspi10)

IP_raspi8=192.168.1.104
IP_raspi9=192.168.1.101
IP_raspi10=192.168.1.103

echo "${name_worker[1]}"
echo "yes"

SRCDIR_root=/Users/qitianyu/Master/Semester1/Federated_learning/
dir1=FL_models
dir2=model
dir3=noniid
dir4=train
dir5=init
file1=main.py

DESDIR=/home/pi/qitianyu/Federated_learning/


#echo "ready to..."&&
#cd ${SRCDIR_root}&&
#echo "******************local file ready!!!*************"&&
#scp -v -r ${dir1} ${USER}@${IP_raspi8}:${DESDIR}&&
#scp -v -r ${dir2} ${USER}@${IP_raspi8}:${DESDIR}&&
#scp -v -r ${dir3} ${USER}@${IP_raspi8}:${DESDIR}&&
#scp -v -r ${dir4} ${USER}@${IP_raspi8}:${DESDIR}&&
#scp -v -r ${dir5} ${USER}@${IP_raspi8}:${DESDIR}&&
#scp ${file1} ${USER}@${IP_raspi8}:${DESDIR}&&
#sleep 1
#
#scp -v -r ${dir1} ${USER}@${IP_raspi9}:${DESDIR}&&
#scp -v -r ${dir2} ${USER}@${IP_raspi9}:${DESDIR}&&
#scp -v -r ${dir3} ${USER}@${IP_raspi9}:${DESDIR}&&
#scp -v -r ${dir4} ${USER}@${IP_raspi9}:${DESDIR}&&
#scp -v -r ${dir5} ${USER}@${IP_raspi9}:${DESDIR}&&
#scp ${file1} ${USER}@${IP_raspi9}:${DESDIR}&&
#
#scp -v -r ${dir1} ${USER}@${IP_raspi10}:${DESDIR}&&
#scp -v -r ${dir2} ${USER}@${IP_raspi10}:${DESDIR}&&
#scp -v -r ${dir3} ${USER}@${IP_raspi10}:${DESDIR}&&
#scp -v -r ${dir4} ${USER}@${IP_raspi10}:${DESDIR}&&
#scp -v -r ${dir5} ${USER}@${IP_raspi10}:${DESDIR}&&
#scp ${file1} ${USER}@${IP_raspi10}:${DESDIR}&&
#echo "yes"
