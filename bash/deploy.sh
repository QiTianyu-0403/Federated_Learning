#!/bin/bash

USER=pi
PASSWORD=raspberry

nameworker=(raspi8 raspi9 raspi10)
IP_raspi8=192.168.1.104
IP_raspi9=192.168.1.101
IP_raspi10=192.168.1.103

SRCDIR_root=/Users/qitianyu/Master/Semester1/Federated_learning/
dir1=FL_models
dir2=model
dir3=noniid
dir4=train
dir5=init
file1=main.py

DESDIR=/home/pi/qitianyu/Federated_learning/
FILENAME=./bash/ip.txt

for_in_file(){
   for ip in `cat $FILENAME`
   do
      echo $ip
      scp -v -r ${dir1} ${USER}@${ip}:${DESDIR}
      scp -v -r ${dir2} ${USER}@${ip}:${DESDIR}
      scp -v -r ${dir3} ${USER}@${ip}:${DESDIR}
      scp -v -r ${dir4} ${USER}@${ip}:${DESDIR}
      scp -v -r ${dir5} ${USER}@${ip}:${DESDIR}
      scp ${file1} ${USER}@${ip}:${DESDIR}
      echo "Deploy for $ip is done"
      sleep 1
   done
}

cd ${SRCDIR_root}&&
echo "Ready to deploy..."&&
sleep 1

for_in_file

echo "All deployment and configuration completed successfully~~~~！"