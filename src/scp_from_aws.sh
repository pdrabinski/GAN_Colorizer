#!/bin/bash
#-- start config
# Local Directory for backups. A date-specific folder is created under this directory for the files.
LD_images=~/Projects/GAN/images
LD_models=~/Projects/GAN/models
# Remote Directory to retrieve. Files are retrieved recursively starting here. Hidden files are included.
# Must be full path, don't use ~ shortcut.
RD_images=/home/ubuntu/GAN_Colorizer/images
RD_models=/home/ubuntu/GAN_Colorizer/models
# Path to SSH ID file (private key)
ID=~/.ssh/my_aws_keypair.pem
# USERname to login as
USER=ubuntu
# HOST to login to
HOST=ec2-52-201-236-103.compute-1.amazonaws.com
#--- end config

scp -r -i $ID $USER@$HOST:$RD_images/. $LD_images/.
scp -r -i $ID $USER@$HOST:$RD_models/. $LD_models/.
ssh -i $ID $USER@$HOST 'rm /home/ubuntu/GAN_Colorizer/images/*'
ssh -i $ID $USER@$HOST 'rm /home/ubuntu/GAN_Colorizer/models/*'
