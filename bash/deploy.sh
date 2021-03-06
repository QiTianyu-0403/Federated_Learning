#!/bin/bash

USER=pi
PASSWORD=raspberry
SEVER=qty-tp

# nameworker=(raspi8 raspi9 raspi10)
# IP_raspi8=192.168.0.104
# IP_raspi9=192.168.0.101:
# IP_raspi10=192.168.0.103

SRCDIR_root=/Users/qitianyu/Master/Semester1/Federated_learning/
dir1=FL_models
dir2=model
dir3=noniid
dir4=train
dir5=init
dir6=bash
file1=main.py

DESDIR=/home/pi/qitianyu/Federated_learning/
DESDIR1=/home/jason/qitianyu/Federated_learning/
SERVER_DESDIR=/home/qty-tp/qitianyu/Federated_learning/
FILENAME=./bash/ip.txt

for_in_file(){

   scp -v -r ${dir1} qty-tp@192.168.0.105:${SERVER_DESDIR}
   scp -v -r ${dir2} qty-tp@192.168.0.105:${SERVER_DESDIR}
   scp -v -r ${dir3} qty-tp@192.168.0.105:${SERVER_DESDIR}
   scp -v -r ${dir4} qty-tp@192.168.0.105:${SERVER_DESDIR}
   scp -v -r ${dir5} qty-tp@192.168.0.105:${SERVER_DESDIR}
   scp -v -r ${dir6} qty-tp@192.168.0.105:${SERVER_DESDIR}
   scp ${file1} qty-tp@192.168.0.105:${SERVER_DESDIR}
   echo "Deploy for 192.168.0.105 is done"
   sleep 1

   # For rapberrys:
   for ip in `cat $FILENAME`
   do
      echo $ip
      scp -v -r ${dir1} ${USER}@${ip}:${DESDIR}
      scp -v -r ${dir2} ${USER}@${ip}:${DESDIR}
      scp -v -r ${dir3} ${USER}@${ip}:${DESDIR}
      scp -v -r ${dir4} ${USER}@${ip}:${DESDIR}
      scp -v -r ${dir5} ${USER}@${ip}:${DESDIR}
      scp -v -r ${dir6} ${USER}@${ip}:${DESDIR}
      scp ${file1} ${USER}@${ip}:${DESDIR}
      echo "Deploy for $ip is done"
      sleep 1
   done

   scp -v -r ${dir1} jason@192.168.0.104:${DESDIR1}
   scp -v -r ${dir2} jason@192.168.0.104:${DESDIR1}
   scp -v -r ${dir3} jason@192.168.0.104:${DESDIR1}
   scp -v -r ${dir4} jason@192.168.0.104:${DESDIR1}
   scp -v -r ${dir5} jason@192.168.0.104:${DESDIR1}
   scp -v -r ${dir6} jason@192.168.0.104:${DESDIR1}
   scp ${file1} jason@192.168.0.104:${DESDIR1}
   echo "Deploy for 192.168.0.104 is done"
   sleep 1

   scp -v -r ${dir1} jason@192.168.0.114:${DESDIR1}
   scp -v -r ${dir2} jason@192.168.0.114:${DESDIR1}
   scp -v -r ${dir3} jason@192.168.0.114:${DESDIR1}
   scp -v -r ${dir4} jason@192.168.0.114:${DESDIR1}
   scp -v -r ${dir5} jason@192.168.0.114:${DESDIR1}
   scp -v -r ${dir6} jason@192.168.0.114:${DESDIR1}
   scp ${file1} jason@192.168.0.114:${DESDIR1}
   echo "Deploy for 192.168.0.114 is done"
   sleep 1

   # # For qty-TP:
   # scp -v -r ${dir1} ${SEVER}@192.168.1.110:${SERVER_DESDIR}
   # scp -v -r ${dir2} ${SEVER}@192.168.1.110:${SERVER_DESDIR}
   # scp -v -r ${dir3} ${SEVER}@192.168.1.110:${SERVER_DESDIR}
   # scp -v -r ${dir4} ${SEVER}@192.168.1.110:${SERVER_DESDIR}
   # scp -v -r ${dir5} ${SEVER}@192.168.1.110:${SERVER_DESDIR}
   # scp -v -r ${dir6} ${SEVER}@192.168.1.110:${SERVER_DESDIR}
   # scp ${file1} ${SEVER}@192.168.1.110:${SERVER_DESDIR}
   # echo "Deploy for Server is done"
   # sleep 1

   # # For qty-desk:
   # scp -v -r bash qty@192.168.1.109:/home/qty/Semester2
   # echo "Deploy for bash is done"
   # sleep 1
}

cd ${SRCDIR_root}&&
echo "Ready to deploy..."&&
sleep 1

for_in_file

echo "All deployment and configuration completed successfully~~~~???"
