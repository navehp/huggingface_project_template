ssh-keygen
# Press enter all the way
for SERVER_NUM in 11 12 13 15 16
do
ssh-copy-id -i ~/.ssh/id_rsa.pub "navehp@nlp${SERVER_NUM}.iem.technion.ac.il"
done