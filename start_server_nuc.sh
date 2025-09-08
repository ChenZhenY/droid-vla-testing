docker stop nuc-setup_nuc-1
sudo pkill -9 run_server

sudo iptables -P INPUT ACCEPT
sudo iptables -P OUTPUT ACCEPT
sudo iptables -P FORWARD ACCEPT
sudo iptables -F
sudo ufw disable

cd /home/rl2-nuc4/droid_workspace/droid/scripts/server
./launch_server.sh