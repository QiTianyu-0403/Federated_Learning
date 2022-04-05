#!/bin/bash

gnome-terminal --window -x bash -c \
"\
ssh -t qty-tp@192.168.1.110 << remotessh
spawn su - root
expect "Password:"
send "qtyszbd\n"
interact
cd Semester2/Federated_learning
sh bash/kill.sh
python main.py --rank 0
exec bash;\
"

gnome-terminal --window -x bash -c \
"\
ssh -t pi@192.168.1.101 << remotessh
cd qitianyu/Federated_learning
sh bash/kill.sh
python main.py --rank 1
exec bash;\
"

gnome-terminal --window -x bash -c \
"\
ssh -t pi@192.168.1.103 << remotessh
cd qitianyu/Federated_learning
sh bash/kill.sh
python main.py --rank 2
exec bash;\
"

gnome-terminal --window -x bash -c \
"\
ssh -t pi@192.168.1.104 << remotessh
cd qitianyu/Federated_learning
sh bash/kill.sh
python main.py --rank 3
exec bash;\
"
